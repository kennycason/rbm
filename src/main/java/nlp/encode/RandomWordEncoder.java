package nlp.encode;

import math.DenseMatrix;
import math.Matrix;

import java.util.Random;

/**
 * Created by kenny on 6/3/14.
 * generate a random vector for a word
 */
public class RandomWordEncoder implements WordEncoder {

    private final static Random RANDOM = new Random();

    private final int dimensions;

    public RandomWordEncoder() {
        this(20);
    }

    public RandomWordEncoder(int dimensions) {
        this.dimensions = dimensions;
    }

    public Matrix encode(String word) {
        Matrix matrix = DenseMatrix.make(1, dimensions);
        for(int i = 0; i < dimensions; i++) {
            matrix.set(0, i, RANDOM.nextDouble());
        }
        return matrix;
    }

}
