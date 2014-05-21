package nn.rbm;

import math.matrix.Matrix;
import math.matrix.MutableMatrix;

/**
 * Created by kenny on 5/12/14.
 *
 */
public class RBM {

    private final int visibleSize;

    private final int hiddenSize;

    private final Matrix weights;

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
