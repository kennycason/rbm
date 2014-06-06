package nlp.tokenizer;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class EnglishSentenceTokenizer {

    public List<List<HasWord>> tokenize(String fileName) {
        final DocumentPreprocessor dp = new DocumentPreprocessor(getReader(fileName));

        final List<List<HasWord>> sentences = new ArrayList<>();
        for (List<HasWord> sentence : dp) {
            sentences.add(sentence);
        }
        return sentences;
    }

    public List<List<HasWord>> tokenizeString(String text) {
        final StringReader stringReader = new StringReader(text);
        final DocumentPreprocessor dp = new DocumentPreprocessor(stringReader);

        final List<List<HasWord>> sentences = new ArrayList<>();
        for (List<HasWord> sentence : dp) {
            sentences.add(sentence);
        }
        return sentences;
    }

    private Reader getReader(final String file) {
        return new InputStreamReader(EnglishSentenceTokenizer.class.getResourceAsStream(file));
    }

}
