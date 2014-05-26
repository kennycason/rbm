package nn.rbm;

import math.DenseMatrix;
import math.Matrix;

import java.util.Random;

/**
 * Created by kenny on 5/12/14.
 *
 */
public class RBM {

    private static final Random RANDOM = new Random();

    private Matrix weights;

    public RBM(final int visibleSize, final int hiddenSize) {
        this.weights = DenseMatrix.randomGaussian(visibleSize, hiddenSize);
    }

    public int getVisibleSize() {
        return weights.rows();
    }

    public int getHiddenSize() {
        return weights.columns();
    }

    public void addVisibleNodes(int n) {

        final Matrix weights = DenseMatrix.make(getVisibleSize() + n, getHiddenSize());
        // copy original values
        for(int i = 0; i < this.weights.rows(); i++) {
            for(int j = 0; j < this.weights.columns(); j++) {
                weights.set(i, j, this.weights.get(i, j));
            }
        }
        // randomly init new weights;
        for(int i = 0; i < this.weights.rows(); i++) {
            for(int j = this.weights.columns(); j < weights.columns(); j++) {
                weights.set(i, j, RANDOM.nextGaussian() * 0.1);
            }
        }
        this.weights = weights;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    @Override
    public String toString() {
        return "RBM{" +
                "visibleSize=" + getVisibleSize() +
                ", hiddenSize=" + getHiddenSize() +
                ", weights=" + weights +
                '}';
    }

}
