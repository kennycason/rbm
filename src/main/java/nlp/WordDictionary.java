package nlp;

import ch.lambdaj.Lambda;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.Morphology;
import math.Matrix;
import math.functions.distance.DistanceFunction;
import math.functions.distance.EuclideanDistanceFunction;
import nlp.encode.DiscreteRandomWordEncoder;
import nlp.encode.WordEncoder;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by kenny on 5/23/14.
 */
public class WordDictionary {

    private static final Logger LOGGER = Logger.getLogger(WordDictionary.class);

    // for getting word stems;
    private static final Morphology MORPHOLOGY = new Morphology();

    private boolean useStems = true;

    // a unique word vector for each word (storing a 1xN matrix)
    private Map<String, Matrix> wordVectors = new HashMap<>();

    private WordEncoder wordEncoder = new DiscreteRandomWordEncoder();

    public WordDictionary() {}

    // load a list of new line separated words
    public WordDictionary(String file) {
        load(file);
    }

    private void load(String file) {
        final List<String> lines = readLines(file);
        for(String line : lines) {
            if(line.startsWith("#")) { continue; } // ignore comments
            add(line);
        }
    }

    public void add(String word) {
        if(useStems) {
            word = String.valueOf(MORPHOLOGY.stem(new Word(word)));
        }
        // LOGGER.info("adding: " + word);
        if(wordVectors.containsKey(word)) { return; }

        wordVectors.put(word, wordEncoder.encode(word));
    }

    public boolean contains(final String word) {
        return wordVectors.containsKey(word);
    }

    public Matrix getVector(String word) {
        if(useStems) {
            word = String.valueOf(MORPHOLOGY.stem(new Word(word)));
        }
        return wordVectors.get(word);
    }

    public List<Matrix> getWordVectors() {
        return new ArrayList<>(wordVectors.values());
    }

    public String buildSentence(final List<Matrix> wordVectors) {
        final List<String> words = new ArrayList<>();
        for(Matrix wordVector : wordVectors) {
            words.add(getClosestWord(wordVector));
        }
        return Lambda.join(words);
    }

    // TODO speed up
    public String getClosestWord(final Matrix wordVector) {
        final DistanceFunction distanceFunction = new EuclideanDistanceFunction();

        String closest = null;
        double minDistance = Double.MAX_VALUE;
        for(Map.Entry<String, Matrix> entry : this.wordVectors.entrySet()) {
            final double distance = distanceFunction.distance(entry.getValue(), wordVector);
            if(distance < minDistance) {
                minDistance = distance;
                closest = entry.getKey();
            }
        }
        return closest;
    }

    public int size() {
        return wordVectors.size();
    }

    private List<String> readLines(final String file) {
        try {
            return IOUtils.readLines(WordDictionary.class.getResourceAsStream(file));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
            return Collections.emptyList();
        }
    }

}
