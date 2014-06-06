package nlp.encode;

import math.DenseMatrix;
import math.Matrix;

import java.util.Random;

/**
 * Created by kenny on 6/3/14.
 * generate a random vector for a word
 */
public class DiscreteRandomWordEncoder implements WordEncoder {

    private final static Random RANDOM = new Random();

    private final int dimensions;

    public DiscreteRandomWordEncoder() {
        this(20);
    }

    public DiscreteRandomWordEncoder(int dimensions) {
        this.dimensions = dimensions;
    }

    public Matrix encode(String word) {
        Matrix matrix = DenseMatrix.make(1, dimensions);
        for(int i = 0; i < dimensions; i++) {
            matrix.set(0, i, RANDOM.nextGaussian() > 0 ? 1.0 : 0.0);
        }
        return matrix;
    }

}
