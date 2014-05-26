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
import nn.rbm.save.DeepRBMPersister;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;
import sentiment.nlp.ngram.NGramGenerator;
import sentiment.nlp.tokenizer.EnglishSentenceTokenizer;
import sentiment.nlp.tokenizer.EnglishWordTokenizer;
import utils.Clock;

import java.io.IOException;
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

    private static final DeepRBMPersister DEEP_RBM_PERSISTER = new DeepRBMPersister();

    private static final int VISIBLE_LAYER_SIZE = 100000;

    private final LearningParameters learningParameters;

    private final MultiThreadedDeepContrastiveDivergence multiThreadedDeepContrastiveDivergence;

    private final IndexedTokenDictionary indexedTokenDictionary;

    private DeepRBM deepRBM;

    public static void main(String[] args) {
        final SentimentClassifier sentimentClassifier = new SentimentClassifier();
        sentimentClassifier.train();
        sentimentClassifier.save("/tmp/sentiment_rbm.csv");

        // positive
        final String sentence = "I absolutely LOVE this stuff. I use it mostly for my sheets, towels and comforters. You can use as much or as little as you want and it just smells wonderful. You walk in your bedroom and it just smells clean! What a great invention! I always buy when I see it on sale! Try it! I think you will love it!";
        LOGGER.info(sentimentClassifier.classify(sentence));

        // negative
        final String sentence2 = "I loved the scent of this, it was on sale & I had a coupon. I'd read reviews about the scent being very strong so used in sparingly in the first load. Nothing. Tried it again and filled the entire cap, as I was doing a big load. Some items had a light pleasant fragrance, others none at all. It seemed that only the all cotton items carried the scent. However it would be far to expensive, in my opinion, to justify using it regularly. And after a few hours, the scent is definitely fading. Oh well! It was fun to try it.";
        LOGGER.info(sentimentClassifier.classify(sentence2));

        // negative
        final String sentence3 = "First of all, let me say I was so excited to try this! Who doesn't like good smelling laundry? I love the smell of the \\\"Fresh Scent\\\" unstopables.I thought the best way to try this out would be using it on blankets, towels and sheets. In the directions it states \\\"use as little or as much as you want\\\" so I decided to try it both ways. I started off using a little amount and thought maybe this was why the smell didn't last very long. Then I used a lot, I was so disappointed, the smell only lasted a couple of days also! My Downy fabric softener lasts longer than this!!! I won't be buying this product again, I will be sticking to my Tide laundry detergent and Downy fabric softener.";
        LOGGER.info(sentimentClassifier.classify(sentence3));

        // postive
        final String sentence4 = "I have these on subscribe and save, very good for things like blankets that are not washed very frequently, freshness lasts";
        LOGGER.info(sentimentClassifier.classify(sentence4));

    }

    public SentimentClassifier() {
        this.learningParameters = new LearningParameters().setEpochs(250).setLog(true);
        this.multiThreadedDeepContrastiveDivergence = new MultiThreadedDeepContrastiveDivergence(learningParameters);
        this.deepRBM = buildDeepRBM();
        this.indexedTokenDictionary = new IndexedTokenDictionary("/data/nlp/english_top100k.txt");
    }

    public void train() {
        train(loadTrainingData("/data/nlp/ecommerce_reviews_100.csv"));
    }

    public void load(final String file) {
        this.deepRBM = DEEP_RBM_PERSISTER.load(file);
    }

    public void save(final String file) {
        DEEP_RBM_PERSISTER.save(this.deepRBM, file);
    }

    public Sentiment classify(final String sentence) {
        LOGGER.info("Classifying: " + sentence);
        final Matrix input = buildInputFromSentence(sentence);
        final Matrix hidden = this.multiThreadedDeepContrastiveDivergence.runVisible(this.deepRBM, input);
        final Matrix visible = this.multiThreadedDeepContrastiveDivergence.runHidden(this.deepRBM, hidden);
        LOGGER.info("Raw Output: " + visible.get(0, VISIBLE_LAYER_SIZE - 2) + " " + visible.get(0, VISIBLE_LAYER_SIZE - 1));
        LOGGER.info("error: " + Vector.getSquaredError(input.row(0).toArray(), visible.row(0).toArray()) / input.columns());

        if(isPositive(visible)) { return Sentiment.POSITIVE; }
        if(isNegative(visible)) { return Sentiment.NEGATIVE; }
        return Sentiment.NEUTRAL;
    }

    private Matrix buildInputFromSentence(final String sentence) {
        final List<CoreLabel> tokens = ENGLISH_WORD_TOKENIZER.tokenize(sentence);
        final double[] data = new double[Math.max(this.indexedTokenDictionary.size(), VISIBLE_LAYER_SIZE)];
        for(CoreLabel token : tokens) {
            final String value = token.value();
            if(!this.indexedTokenDictionary.contains(value)) { continue; }

            final int index = this.indexedTokenDictionary.index(value);
            data[index] = 1.0;
        }
        return DenseMatrix.make(new double[][]{data});
    }

    /*
     * load csv, first column = 1-5 rating, second column is comment
     */
    private Matrix loadTrainingData(final String file) {
        LOGGER.info("Loading training data from: " + file);
        try {
            final List<String> lines = IOUtils.readLines(SentimentClassifier.class.getResourceAsStream(file));

            final Matrix[] trainingData = new Matrix[lines.size()];
            int i = 0;
            for(String line : lines) {
                final int firstCommaIndex = line.indexOf(',');
                final double rating = Double.parseDouble(line.substring(0, firstCommaIndex));
                final String comment = line.substring(firstCommaIndex + 2, line.length() - 1); // trim quotes

                final Matrix input = this.buildInputFromSentence(comment);
                insertTrainingLabels(input, rating);
                trainingData[i] = input; // todo parse to sentences
                i++;
            }

            return DenseMatrix.make(Matrix.concatRows(trainingData));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }

        return DenseMatrix.make(0, Math.max(this.indexedTokenDictionary.size(), VISIBLE_LAYER_SIZE));
    }

    private void insertTrainingLabels(Matrix input, double rating) {
        input.set(0, VISIBLE_LAYER_SIZE - 2, 0.0);
        input.set(0, VISIBLE_LAYER_SIZE - 1, 0.0);
        if(rating > 3) {
            input.set(0, VISIBLE_LAYER_SIZE - 2, 1.0);
        }
        else if(rating < 3) {
            input.set(0, VISIBLE_LAYER_SIZE - 1, 1.0);
        }
    }

    private boolean isPositive(Matrix input) {
        return input.get(0, VISIBLE_LAYER_SIZE - 2) > 0.7;
    }

    private boolean isNegative(Matrix input) {
        return input.get(0, VISIBLE_LAYER_SIZE - 1) > 0.7;
    }

    private DeepRBM buildDeepRBM() {
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(200).setVisibleUnitsPerRBM(500).setHiddenUnitsPerRBM(100),    // 100,000 in, 20,000 out
//                new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 20,000 in, 5,000 out
//                new LayerParameters().setNumRBMS(50).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 5,000 in, 500 out
//                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 500 in, 250 out
//                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 250 in, 50 out
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(100),    // 50 in, 100 out
//        };

        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(500).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(25),    // 100,000 in, 12,500 out
                new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(125).setHiddenUnitsPerRBM(15),    // 12,500 in, 1,500 out
                new LayerParameters().setNumRBMS(20).setVisibleUnitsPerRBM(75).setHiddenUnitsPerRBM(10),     // 1,500 in, 200 out
                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(5),      // 500 in, 50 out
                new LayerParameters().setNumRBMS(5).setVisibleUnitsPerRBM(10).setHiddenUnitsPerRBM(2),       // 50 in, 10 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(10).setHiddenUnitsPerRBM(20),     // 10 in 20 out
        };

        final DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);

        return deepRBM;
    }

    private void train(final Matrix trainingData) {
        LOGGER.info("Start training");
        final Clock clock = new Clock();
        clock.start();
        LOGGER.info("training data size: " + trainingData.rows());
        multiThreadedDeepContrastiveDivergence.learn(deepRBM, trainingData);
        final long elapsedMilliseconds = clock.elapsedMillis();
        LOGGER.info("Finished training in " + elapsedMilliseconds + "ms");
    }


}
