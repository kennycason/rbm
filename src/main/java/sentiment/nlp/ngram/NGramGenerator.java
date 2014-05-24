package sentiment.nlp.ngram;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class NGramGenerator {

    private final int n;

    public NGramGenerator(final int n) {
        this.n = n;
    }

    public List<String> generate(String sentence) {
        final List<String> ngrams = new ArrayList<String>();
        String[] words = sentence.split(" ");
        for (int i = 0; i < words.length - n + 1; i++)
            ngrams.add(concat(words, i, i + n));
        return ngrams;
    }

    public String concat(String[] words, int start, int end) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end; i++)
            sb.append((i > start ? " " : "") + words[i]);
        return sb.toString();
    }

}
