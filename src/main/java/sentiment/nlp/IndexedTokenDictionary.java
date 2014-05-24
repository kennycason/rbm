package sentiment.nlp;

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
public class IndexedTokenDictionary {

    private static final Logger LOGGER = Logger.getLogger(IndexedTokenDictionary.class);

    private Map<String, Integer> tokenIndex = new HashMap<>();

    private List<String> tokens = new ArrayList<>(100000);

    public IndexedTokenDictionary(String file) {
        load(file);
    }

    private void load(String file) {
        final List<String> lines = readLines(file);
        int i = 0;
        for(String line : lines) {
            if(line.startsWith("#")) { continue; }
            if(tokenIndex.containsKey(line)) { continue; }

            tokenIndex.put(line, i);
            tokens.add(line);
            i++;
        }
    }

    public boolean contains(String token) {
        return tokenIndex.containsKey(token);
    }

    public Integer index(String token) {
        return tokenIndex.get(token);
    }

    public int size() {
        return tokenIndex.size();
    }

    private List<String> readLines(final String file) {
        try {
            return IOUtils.readLines(IndexedTokenDictionary.class.getResourceAsStream(file));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
            return Collections.emptyList();
        }
    }

}
