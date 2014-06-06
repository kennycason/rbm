package nlp.tokenizer;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;

import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class EnglishWordTokenizer {

    public List<CoreLabel> tokenizeFromFile(String fileName) {
        final PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer(getReader(fileName), new CoreLabelTokenFactory(), "");

        final List<CoreLabel> words = new ArrayList<>();
        while(ptbt.hasNext()) {
            words.add(ptbt.next());
        }
        return words;
    }

    public List<CoreLabel> tokenize(String sentence) {
        final PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer(new StringReader(sentence), new CoreLabelTokenFactory(), "");


        final List<CoreLabel> words = new ArrayList<>();
        for(CoreLabel word : ptbt.tokenize()) {

            words.add(word);
        }
        return words;
    }

    private Reader getReader(final String file) {
        return new InputStreamReader(
                EnglishWordTokenizer.class.getResourceAsStream(file));
    }

}
