package nn.rbm;

import math.matrix.Matrix;
import math.matrix.MutableMatrix;

import java.util.Random;

/**
 * Created by kenny on 5/12/14.
 *
 */
public class RBM {

    private static final Random RANDOM = new Random();

    private final int visibleSize;

    private final int hiddenSize;

    private Matrix weights;

    public RBM(final int visibleSize, final int hiddenSize) {
        this.visibleSize = visibleSize;
        this.hiddenSize = hiddenSize;
        this.weights = new MutableMatrix(new double[visibleSize][hiddenSize]);
    }

    public int getVisibleSize() {
        return visibleSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void addVisibleNodes(int n) {
        final Matrix weights = new MutableMatrix(visibleSize + n, hiddenSize);
        // copy original values
        for(int i = 0; i < this.weights.rows(); i++) {
            for(int j = 0; j < this.weights.cols(); j++) {
                weights.set(i, j, this.weights.get(i, j));
            }
        }
        // randomly init new weights;
        for(int i = 0; i < this.weights.rows(); i++) {
            for(int j = this.weights.cols(); j < weights.cols(); j++) {
                weights.set(i, j, RANDOM.nextGaussian() * 0.1);
            }
        }
        this.weights = weights;
    }

    public Matrix getWeights() {
        return weights;
    }

    @Override
    public String toString() {
        return "RBM{" +
                "visibleSize=" + visibleSize +
                ", hiddenSize=" + hiddenSize +
                ", weights=" + weights +
                '}';
    }

}
