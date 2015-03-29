package nn.rbm.learn;

import math.DenseMatrix;
import math.Matrix;
import nlp.WordDictionary;
import nn.rbm.RBM;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RBMFactory;
import nn.rbm.factory.RandomRBMFactory;
import org.apache.log4j.Logger;
import org.junit.Test;

import java.util.List;

/**
 * Created by kenny on 6/4/14.
 *
 *
 */
public class TestLearnWords {

    private static final Logger LOGGER = Logger.getLogger(TestLearnWords.class);

    private static final RBMFactory RBM_FACTORY = new RandomRBMFactory();

    private static final WordDictionary WORD_DICTIONARY = new WordDictionary("/data/nlp/english_top100k.txt");

    @Test
    public void learnWords() {
        runExperiment(100, 1);    // 100.0% (100 hidden)
        runExperiment(1000, 10);    // 100.0% (100 hidden)
        //runExperiment(10000, 100);    // 99.0% (100 hidden)
        //runExperiment(20000, 1000);   // 96.0% (100 hidden)
        //runExperiment(30000, 10000);    // 94.7%% (100 hidden)
    }

    @Test
    public void learnWordsDeep() {
        runExperimentDeep(10000, 100);
    }

    public void runExperiment(final int epochs, final int numberWords) {
        final LearningParameters learningParameters = new LearningParameters().setEpochs(epochs);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(learningParameters);
        final RBM rbm = RBM_FACTORY.build(20, 100);

        final List<Matrix> words = WORD_DICTIONARY.getWordVectors().subList(0, numberWords);
        final Matrix trainingData = DenseMatrix.make(Matrix.concatRows(words));

        LOGGER.info("Start Learning");
        contrastiveDivergence.learn(rbm, trainingData);

        LOGGER.info("Measure Success");
        double correct = 0;
        int i = 0;
        for(Matrix word : words) {
            final String actualWord = WORD_DICTIONARY.getClosestWord(word);
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, word);
            final Matrix visible = contrastiveDivergence.runHidden(rbm, hidden);
            final String predictedWord = WORD_DICTIONARY.getClosestWord(visible);
            LOGGER.info(i + ": actual [" + actualWord + "] => predicted [" + predictedWord + "]");

            if(actualWord.equals(predictedWord)) {
                correct++;
            }
            i++;
        }
        LOGGER.info("Success Rate: " + (correct / words.size() * 100.0) + "%");
    }

    public void runExperimentDeep(final int epochs, final int numberWords) {
        final LearningParameters learningParameters = new LearningParameters().setEpochs(epochs);
        final MultiThreadedDeepContrastiveDivergence contrastiveDivergence = new MultiThreadedDeepContrastiveDivergence(learningParameters);

        // 0.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(2).setHiddenUnitsPerRBM(5),    // 20 -> 50
//                new LayerParameters().setNumRBMS(5).setVisibleUnitsPerRBM(10).setHiddenUnitsPerRBM(5),    // 50 -> 25
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(30)    // 25 -> 30
//        };

        // config gives 3.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(4).setVisibleUnitsPerRBM(5).setHiddenUnitsPerRBM(10),    // 20 -> 40
//                new LayerParameters().setNumRBMS(2).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),   // 40 -> 20
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(30)    // 20 -> 30
//        };

        // 24.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(30),    // 20 -> 30
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(30).setHiddenUnitsPerRBM(20),   // 30 -> 20
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(50)    // 20 -> 30
//        };

        // 5%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(30),    // 20 -> 30
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(30).setHiddenUnitsPerRBM(20),   // 30 -> 20
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10)    // 20 -> 10
//        };

        // 54.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(30),    // 20 -> 30
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(30).setHiddenUnitsPerRBM(50),   // 30 -> 50
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(80)    // 50 -> 80
//        };

        // 85.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(50),    // 20 -> 50
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(100),   // 50 -> 100
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(150)    // 100 -> 150
//        };

          // 92.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(70),    // 20 -> 70
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(70).setHiddenUnitsPerRBM(140),   // 70 -> 140
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(140).setHiddenUnitsPerRBM(200)    // 140 -> 200
//        };

        // 90.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(100),    // 20 -> 100
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(200),   // 100 -> 200
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(400)    // 200 -> 400
//        };

        // 91.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(50),    // 20 -> 50
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(150),   // 50 -> 150
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(150).setHiddenUnitsPerRBM(300)    // 150 -> 300
//        };

        // 63.0%
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(2).setHiddenUnitsPerRBM(10),    // 20 -> 100
//                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(10).setHiddenUnitsPerRBM(30),    // 100 -> 300
//                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(30).setHiddenUnitsPerRBM(60)    // 300 -> 600
//        };

        // 87.0%
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(70),    // 20 -> 70
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(70).setHiddenUnitsPerRBM(120),   // 70 -> 120
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(120).setHiddenUnitsPerRBM(300)    // 120 -> 300
        };

        final DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);

        final List<Matrix> words = WORD_DICTIONARY.getWordVectors().subList(0, numberWords);
        final Matrix trainingData = DenseMatrix.make(Matrix.concatRows(words));

        LOGGER.info("Start Learning");
        contrastiveDivergence.learn(deepRBM, trainingData);

        LOGGER.info("Measure Success");
        double correct = 0;
        int i = 0;
        for(Matrix word : words) {
            final String actualWord = WORD_DICTIONARY.getClosestWord(word);
            final Matrix hidden = contrastiveDivergence.runVisible(deepRBM, word);
            final Matrix visible = contrastiveDivergence.runHidden(deepRBM, hidden);
            final String predictedWord = WORD_DICTIONARY.getClosestWord(visible);
            LOGGER.info(i + ": actual [" + actualWord + "] => predicted [" + predictedWord + "]");

            if(actualWord.equals(predictedWord)) {
                correct++;
            }
            i++;
        }
        LOGGER.info("Success Rate: " + (correct / words.size() * 100.0) + "%");
    }

}
