package nlp;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.Morphology;
import math.DenseMatrix;
import math.Matrix;
import math.functions.Sigmoid;
import math.functions.doubledouble.rbm.ActivationState;
import nlp.tokenizer.EnglishSentenceTokenizer;
import nlp.tokenizer.EnglishWordTokenizer;
import nn.rbm.RBM;
import nn.rbm.factory.RBMFactory;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.learn.LearningParameters;
import nn.rbm.learn.RecurrentContrastiveDivergence;
import nn.rbm.save.RBMPersister;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;
import utils.Clock;
import utils.PrettyPrint;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by kenny on 5/23/14.
 * experimental
 * word vectors [0..19 word vector|2 bit +/- label]
 *
 */
public class SentimentClassifierRBM {

    private static final Logger LOGGER = Logger.getLogger(SentimentClassifierRBM.class);

    private static final EnglishSentenceTokenizer ENGLISH_SENTENCE_TOKENIZER = new EnglishSentenceTokenizer();
    private static final EnglishWordTokenizer ENGLISH_WORD_TOKENIZER = new EnglishWordTokenizer();
    private static final RBMFactory RBM_FACTORY = new RandomRBMFactory();
    private static final RBMPersister RBM_PERSISTER = new RBMPersister();
    private static final Morphology MORPHOLOGY = new Morphology();

    private static final DoubleFunction SIGMOID = new Sigmoid();
    private static final DoubleDoubleFunction ACTIVATION_STATE = new ActivationState();

    private final LearningParameters learningParameters;
    private final RecurrentContrastiveDivergence contrastiveDivergence;
    private final WordDictionary wordDictionary;

    private RBM rbm;

    private static Map<String, Double> reviewRatings = new HashMap<>();

    public static void main(String[] args) {
        final SentimentClassifierRBM sentimentClassifier = new SentimentClassifierRBM();
        sentimentClassifier.train();
        sentimentClassifier.save("/tmp/sentiment_rbm_stem.csv");

        int correct = 0;
        for(Map.Entry<String, Double> reviewRating : reviewRatings.entrySet()) {
            final Sentiment predicted = sentimentClassifier.classify(reviewRating.getKey());
            if(reviewRating.getValue() > 3 && predicted == Sentiment.POSITIVE) {
                correct ++;
            }
            if(reviewRating.getValue() < 3 && predicted == Sentiment.NEGATIVE) {
                correct ++;
            }
            LOGGER.info("Rating should be (" + reviewRating.getValue() + ") = " + predicted);
        }

        LOGGER.info(correct / (double) reviewRatings.size() * 100.0 + "% correct");
    }

    public SentimentClassifierRBM() {
        this.learningParameters = new LearningParameters().setEpochs(500).setLog(true).setMemory(5);
        this.contrastiveDivergence = new RecurrentContrastiveDivergence(learningParameters);
        this.rbm = buildRBM();
        this.wordDictionary = new WordDictionary("/data/nlp/english_top100k.txt");
    }

    public void train() {
        train(loadTrainingData("/data/nlp/ecommerce_reviews_100.csv"));
    }

    public void load(final String file) {
        this.rbm = RBM_PERSISTER.load(file);
    }

    public void save(final String file) {
        RBM_PERSISTER.save(this.rbm, file);
    }

    public Sentiment classify(final String text) {
        LOGGER.info("Classifying: " + text);
        final List<List<Matrix>> sentences = parseTextIntoSentences(text);

        int positive = 0;
        int negative = 0;
        for(List<Matrix> sentence : sentences) {
            final Matrix visible = visualizeEvents(this.rbm, sentence);
            LOGGER.info(visible);
            LOGGER.info("Raw Output: " + PrettyPrint.toPixelBox(visible.row(0).toArray(), visible.columns(), 0.5));

            if(isPositive(visible) && isNegative(visible)) { continue; }
            if(isPositive(visible)) { positive++; }
            if(isNegative(visible)) { negative++; }
        }
        if(positive > negative) { return Sentiment.POSITIVE; }
        if(negative > positive) { return Sentiment.NEGATIVE; }
        return Sentiment.NEUTRAL;
    }

    /*
     * specialized version of Recurrent Contrastive Divergence's version that propagates predicted classification signal
     *
     */
    private Matrix visualizeEvents(RBM rbm, List<Matrix> sentence) {
        final Matrix weights = rbm.getWeights();

        Matrix lastVisibleStates;

        double lastPositive = 0.0;
        double lastNegative = 0.0;
        int event = 0;
        do {
            insertTrainingLabels(sentence, lastPositive , lastNegative);
            final Matrix currentAndNextEvent = createTemporalInput(event, sentence);

            // run visible
            // Calculate the activations of the hidden units.
            final Matrix hiddenActivations = currentAndNextEvent.dot(weights);
            // Calculate the probabilities of turning the hidden units on.
            final Matrix hiddenProbabilities = hiddenActivations.apply(SIGMOID);
            // Turn the hidden units on with their specified probabilities.
            final Matrix hiddenStates = hiddenProbabilities.apply(DenseMatrix.random(currentAndNextEvent.rows(), rbm.getHiddenSize()), ACTIVATION_STATE);

            // run hidden
            // Calculate the activations of the hidden units.
            final Matrix visibleActivations = hiddenStates.dot(weights.transpose());
            // Calculate the probabilities of turning the visible units on.
            final Matrix visibleProbabilities = visibleActivations.apply(SIGMOID);
            // Turn the visible units on with their specified probabilities.
            final Matrix visibleStates = visibleProbabilities.apply(DenseMatrix.random(hiddenStates.rows(), rbm.getVisibleSize()), ACTIVATION_STATE);

            lastVisibleStates = visibleStates;

            event++;

            // look at last visible state's sentiment prediction
            if(isPositive(visibleStates) && isNegative(visibleStates)) { continue; }
            if(isPositive(visibleStates)) { lastPositive = visibleStates.get(0, positiveIndex(visibleStates)); }
            if(isNegative(visibleStates)) { lastNegative = visibleStates.get(0, negativeIndex(visibleStates)); }

        } while(event < sentence.size() - learningParameters.getMemory());

        return lastVisibleStates;
    }


    private Matrix createTemporalInput(int event, List<Matrix> events) {
        final Matrix currentEvent = events.get(event);

        Matrix temporalEvent = currentEvent;
        for(int i = event + 1, t = 0; i < events.size() && t < learningParameters.getMemory(); i++, t++) {
            temporalEvent = temporalEvent.addColumns(events.get(i));
        }

        final int temporalEventColumns = currentEvent.columns() + currentEvent.columns() * learningParameters.getMemory();
        if(temporalEvent.columns() < temporalEventColumns) { // fill in blanks if there is not enough temporal data to train
            temporalEvent = temporalEvent.addColumns(DenseMatrix.make(currentEvent.rows(), temporalEventColumns - temporalEvent.columns()));
        }
        return temporalEvent;
    }

    private List<List<Matrix>> parseTextIntoSentences(final String text) {
        List<List<Matrix>> sentences = new ArrayList<>();
        List<List<HasWord>> parsedSentences = ENGLISH_SENTENCE_TOKENIZER.tokenizeString(text);

        for(List<HasWord> parsedSentence : parsedSentences) {
            final List<Matrix> sentence = new ArrayList<>(parsedSentence.size());
            for(HasWord token : parsedSentence) {
                final String word = String.valueOf(MORPHOLOGY.stem(token.word()));
                if(!this.wordDictionary.contains(word)) { continue; }

                final Matrix wordVector = this.wordDictionary.getVector(word);
                final Matrix wordVectorWithLabelSpace = wordVector.addColumns(DenseMatrix.make(1, 2));
                sentence.add(wordVectorWithLabelSpace);
            }
            sentences.add(sentence);
        }
        return sentences;
    }

    private List<List<Matrix>> parseTextIntoSentence(final String text) {
        List<CoreLabel> words = ENGLISH_WORD_TOKENIZER.tokenize(text);

        final List<Matrix> sentence = new ArrayList<>(words.size());
        for(CoreLabel token : words) {
            final String word = String.valueOf(MORPHOLOGY.stem(token.word()));
            if(!this.wordDictionary.contains(word)) { continue; }

            final Matrix wordVector = this.wordDictionary.getVector(word);
            final Matrix wordVectorWithLabelSpace = wordVector.addColumns(DenseMatrix.make(1, 2));
            sentence.add(wordVectorWithLabelSpace);
        }

        return Arrays.asList(sentence);
    }

    /*
     * load csv, first column = 1-5 rating, second column is comment
     */
    private List<List<Matrix>> loadTrainingData(final String file) {
        LOGGER.info("Loading training data from: " + file);
        try {
            final List<String> lines = IOUtils.readLines(SentimentClassifierRBM.class.getResourceAsStream(file)).subList(0, 20);  // select how many to learn
            Collections.sort(lines);

            final List<List<Matrix>> trainingData = new ArrayList<>();
            for(String line : lines) {
                final int firstCommaIndex = line.indexOf(',');
                final double rating = Double.parseDouble(line.substring(0, firstCommaIndex));
                final String text = line.substring(firstCommaIndex + 2, line.length() - 1); // trim quotes
                this.reviewRatings.put(text, rating);
                LOGGER.info(rating + ": " + text);

                final List<List<Matrix>> sentences = this.parseTextIntoSentence(text);
                for(List<Matrix> sentence : sentences) {
                    insertTrainingLabels(sentence, rating);
                    trainingData.add(sentence);
                }
            }
            return trainingData;
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }

        return Collections.emptyList();
    }

    private void insertTrainingLabels(List<Matrix> sentence, double rating) {
        if(rating > 3 && rating <= 5) {
            insertTrainingLabels(sentence, 1.0, 0.0);
        }
        else if(rating < 3 && rating > 0) {
            insertTrainingLabels(sentence, 0.0, 1.0);
        }
    }

    private void insertTrainingLabels(List<Matrix> sentence, double positive, double negative) {
        for(Matrix word : sentence) {
                word.set(0, word.columns() - 1, positive);
                word.set(0, word.columns() - 2, negative);
        }
    }

    private boolean isNegative(Matrix visible) {
        return visible.get(0, negativeIndex(visible)) > 0.7 ? true : false;
    }

    private boolean isPositive(Matrix visible) {
        return visible.get(0, positiveIndex(visible)) > 0.7 ? true : false;
    }

    private int positiveIndex(Matrix visible) {
        return visible.columns() - 1;
    }

    private int negativeIndex(Matrix visible) {
        return visible.columns() - 2;
    }

    private RBM buildRBM() {
        return RBM_FACTORY.build((20 + 2) * 6, 300);
    }

    private void train(final List<List<Matrix>> trainingData) {
        LOGGER.info("Start training");
        LOGGER.info(rbm);
        final Clock clock = new Clock();
        clock.start();
        LOGGER.info("training data size: " + trainingData.size());
        contrastiveDivergence.learnMany(rbm, trainingData);
        final long elapsedMilliseconds = clock.elapsedMillis();
        LOGGER.info("Finished training in " + elapsedMilliseconds + "ms");
    }

}
