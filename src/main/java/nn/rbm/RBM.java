package nn.rbm;

import math.matrix.Matrix;
import math.matrix.MutableMatrix;

/**
 * Created by kenny on 5/12/14.
 *
 */
public class RBM {

    private Layer visible;

    private Layer hidden;

    private Matrix weights;

    public RBM(final Layer visible, final Layer hidden) {
        this.visible = visible;
        this.hidden = hidden;
        this.weights = new MutableMatrix(new double[visible.getSize()][hidden.getSize()]);
    }

    public Layer getVisible() {
        return visible;
    }

    public Layer getHidden() {
        return hidden;
    }

    public Matrix getWeights() {
        return weights;
    }

    @Override
    public String toString() {
        return "RBM{" +
                "visible=" + visible +
                ", hidden=" + hidden +
                ", weights=" + weights +
                '}';
    }

}
