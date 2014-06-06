package nlp.encode;

import math.Matrix;

/**
 * Created by kenny on 6/3/14.
 * TODO create word encoder that captures
 */
public interface WordEncoder {
    Matrix encode(String word);
}
