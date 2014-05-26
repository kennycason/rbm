package sentiment.nlp;

import edu.stanford.nlp.ling.CoreLabel;
import math.DenseMatrix;
import math.Matrix;
import math.Vector;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RBMFactory;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.learn.LearningParameters;
import nn.rbm.learn.MultiThreadedDeepContrastiveDivergence;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;
import sentiment.nlp.ngram.NGramGenerator;
import sentiment.nlp.tokenizer.EnglishSentenceTokenizer;
import sentiment.nlp.tokenizer.EnglishWordTokenizer;
import utils.Clock;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class SentimentClassifier {

    private static final Logger LOGGER = Logger.getLogger(SentimentClassifier.class);

    private static final EnglishSentenceTokenizer ENGLISH_SENTENCE_TOKENIZER = new EnglishSentenceTokenizer();

    private static final EnglishWordTokenizer ENGLISH_WORD_TOKENIZER = new EnglishWordTokenizer();

    private static final NGramGenerator N_GRAM_GENERATOR = new NGramGenerator(3);

    private static final RBMFactory RBM_FACTORY = new RandomRBMFactory();

    private final LearningParameters learningParameters;

    private final MultiThreadedDeepContrastiveDivergence multiThreadedDeepContrastiveDivergence;

    private final IndexedTokenDictionary indexedTokenDictionary;

    private final DeepRBM deepRBM;

    public static void main(String[] args) {
        final SentimentClassifier sentimentClassifier = new SentimentClassifier();
         sentimentClassifier.classify("I am a piece of poop");

    }

    public SentimentClassifier() {
        this.learningParameters = new LearningParameters().setEpochs(100).setLog(false);
        this.multiThreadedDeepContrastiveDivergence = new MultiThreadedDeepContrastiveDivergence(learningParameters);

        this.indexedTokenDictionary = new IndexedTokenDictionary("/data/nlp/english_top100k.txt");
        this.deepRBM = buildDeepRBM();
        //train(loadTrainingData("/data/nlp/ecommerce_reviews_100.csv"));
        train(buildInputFromSentence("I am a piece of poop"));
    }


    public void classify(final String sentence) {
        final Matrix input = buildInputFromSentence(sentence);
        LOGGER.info("input: " + input);
        final Matrix hidden = this.multiThreadedDeepContrastiveDivergence.runVisible(this.deepRBM, input);
        final Matrix visible = this.multiThreadedDeepContrastiveDivergence.runHidden(this.deepRBM, hidden);
        LOGGER.info("output: " + visible);

        LOGGER.info("error: " + Vector.getSquaredError(input.row(0).toArray(), visible.row(0).toArray()) / input.columns());
    }

    private Matrix buildInputFromSentence(final String sentence) {
        final List<CoreLabel> tokens = ENGLISH_WORD_TOKENIZER.tokenize(sentence);
        final double[] data = new double[Math.max(this.indexedTokenDictionary.size(), 100000)];
        for(CoreLabel token : tokens) {
            final String value = token.value();
            if(!this.indexedTokenDictionary.contains(value)) { continue; }

            final int index = this.indexedTokenDictionary.index(value);
            data[index] = 1.0;
        }
        return DenseMatrix.make(new double[][] { data });
    }

    /*
     * load csv, first column = 1-5 rating, second column is comment
     */
    private Matrix loadTrainingData(final String file) {
        LOGGER.info("Loading training data from: " + file);
        try {
            final List<String> lines = IOUtils.readLines(SentimentClassifier.class.getResourceAsStream(file));
            final List<Matrix> trainingData = new ArrayList<>(lines.size());
            for(String line : lines) {
                final int firstCommaIndex = line.indexOf(','); // todo prepend to visual input
                final String rating = line.substring(0, firstCommaIndex);
                final String comment = line.substring(firstCommaIndex + 2, line.length() - 1); // trim quotes
                trainingData.add(this.buildInputFromSentence(comment)); // todo parse to sentences
            }
            return trainingData.get(0); // TODO create concat rows method
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }

        return DenseMatrix.make(0, Math.max(this.indexedTokenDictionary.size(), 100000));
    }

    private DeepRBM buildDeepRBM() {
        // 100 * 63 * 24 input (151200)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(200).setVisibleUnitsPerRBM(500).setHiddenUnitsPerRBM(100),    // 100,000 in, 20,000 out
                new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 20,000 in, 5,000 out
                new LayerParameters().setNumRBMS(50).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 5,000 in, 500 out
                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 500 in, 250 out
                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 250 in, 50 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(100),    // 50 in, 100 out
        };

        final DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);

        return deepRBM;
    }

    private void train(final Matrix trainingData) {
        LOGGER.info("Start training");
        final Clock clock = new Clock();
        clock.start();
        LOGGER.info("input size: " + trainingData.columns() + ", rbm visual size: " + this.deepRBM.getVisibleSize());
        multiThreadedDeepContrastiveDivergence.learn(deepRBM, trainingData);
        final long elapsedMilliseconds = clock.elapsedMillis();
        LOGGER.info("Finished training in " + elapsedMilliseconds + "ms");
    }


}
