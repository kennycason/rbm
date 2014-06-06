package nn.rbm.learn;

import ch.lambdaj.Lambda;
import ch.lambdaj.function.convert.Converter;
import math.DenseMatrix;
import math.Matrix;
import nlp.WordDictionary;
import nn.rbm.RBM;
import nn.rbm.factory.RBMFactory;
import nn.rbm.factory.RandomRBMFactory;
import org.apache.log4j.Logger;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Created by kenny on 6/4/14.
 *
 *
 */
public class TestRecurrentSentence {

    private static final Logger LOGGER = Logger.getLogger(TestRecurrentSentence.class);

    private static final RBMFactory RBM_FACTORY = new RandomRBMFactory();

    private static final WordDictionary WORD_DICTIONARY = new WordDictionary("/data/nlp/english_top100k.txt");

    @Test
    public void recurrentSentence() {
        final LearningParameters learningParameters = new LearningParameters().setEpochs(10000).setLearningRate(0.1).setMemory(3);
        final RecurrentContrastiveDivergence contrastiveDivergence = new RecurrentContrastiveDivergence(learningParameters);
        final RBM rbm = RBM_FACTORY.build(20 + (20 * learningParameters.getMemory()), 100);

        final List<String> words = Arrays.asList(new String[] {
                "I", "am", "an", "american", "living", "in", "china"
        });

        final List<Matrix> sentence = filterNulls(Lambda.convert(words, WORD_TO_VECTOR_CONVERTER));
        contrastiveDivergence.learn(rbm, sentence);


        // walk through all cycles
        LOGGER.info("Predicted Sentence: ");
        for(int i = 0; i < sentence.size() - learningParameters.getMemory(); i++) {
            LOGGER.info("input: " + WORD_DICTIONARY.getClosestWord(sentence.get(i)));

            final Matrix hidden = contrastiveDivergence.runVisible(rbm, sentence.get(i));
            final Matrix visible = contrastiveDivergence.runHidden(rbm, hidden);

            final Matrix prediction = DenseMatrix.make(visible.data().viewPart(0, 20, 1, learningParameters.getMemory() * 20));

            for(int j = 0; j < learningParameters.getMemory(); j++) {
                final String word = WORD_DICTIONARY.getClosestWord(DenseMatrix.make(prediction.data().viewPart(0, j * 20, 1, 20)));
                LOGGER.info("\t\t" + word);
            }
        }

        // short hand to get last predicted value from above
        final Matrix lastVisible = contrastiveDivergence.visualizeEvents(rbm, sentence);
        final String word = WORD_DICTIONARY.getClosestWord(DenseMatrix.make(lastVisible.data().viewPart(0, learningParameters.getMemory() * 20, 1, 20)));
        LOGGER.info("Last predicted: " + word);
    }

    @Test
    public void recurrentSentences() {
        final LearningParameters learningParameters = new LearningParameters().setEpochs(10000).setLearningRate(0.1).setMemory(5);
        final RecurrentContrastiveDivergence contrastiveDivergence = new RecurrentContrastiveDivergence(learningParameters);
        final RBM rbm = RBM_FACTORY.build(20 + (20 * learningParameters.getMemory()), 100);


        final List<String> words = Arrays.asList(new String[] {
                "I", "am", "an", "american", "living", "in", "china"
        });
        final List<String> words2 = Arrays.asList(new String[] {
                "I", "love", "eating", "bananas", "from", "america"
        });

        final List<Matrix> sentence1 = filterNulls(Lambda.convert(words, WORD_TO_VECTOR_CONVERTER));
        final List<Matrix> sentence2 = filterNulls(Lambda.convert(words2, WORD_TO_VECTOR_CONVERTER));

        final List<List<Matrix>> sentences = new ArrayList<>();
        sentences.add(sentence1);
        sentences.add(sentence2);

        contrastiveDivergence.learnMany(rbm, sentences);

        // walk through all cycles
        for(List<Matrix> sentence : sentences) {
            LOGGER.info("Predicted Sentence: ");
            for(int i = 0; i < sentence.size(); i++) {
                LOGGER.info("input: " + WORD_DICTIONARY.getClosestWord(sentence.get(i)));

                final Matrix hidden = contrastiveDivergence.runVisible(rbm, sentence.get(i));
                final Matrix visible = contrastiveDivergence.runHidden(rbm, hidden);

                final Matrix prediction = DenseMatrix.make(visible.data().viewPart(0, 20, 1, learningParameters.getMemory() * 20));

                for(int j = 0; j < learningParameters.getMemory(); j++) {
                    final String word = WORD_DICTIONARY.getClosestWord(DenseMatrix.make(prediction.data().viewPart(0, j * 20, 1, 20)));
                    LOGGER.info("\t\t" + word);
                }
            }

            // short hand to get last predicted value from above
            final Matrix lastVisible = contrastiveDivergence.visualizeEvents(rbm, sentence);
            final String word = WORD_DICTIONARY.getClosestWord(DenseMatrix.make(lastVisible.data().viewPart(0, learningParameters.getMemory() * 20, 1, 20)));
            LOGGER.info("Last predicted: " + word);
        }
    }

    private List<Matrix> filterNulls(List<Matrix> words) {
        Iterator<Matrix> iterator = words.iterator();
        while(iterator.hasNext()) {
            if(iterator.next() == null) {
                iterator.remove();
            }
        }
        return words;
    }

    private static final Converter<String, Matrix> WORD_TO_VECTOR_CONVERTER = new Converter<String, Matrix>() {

        @Override
        public Matrix convert(String word) {
            return WORD_DICTIONARY.getVector(word);
        }
    };

}
