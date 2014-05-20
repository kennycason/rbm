package nn.rbm;

import java.util.Arrays;

/**
 * Created by kenny on 5/12/14.
 */
public class Layer {

    private final int size;

    private double[] values;

    public Layer(final int size) {
        this.size = size;
        this.values = new double[size];
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        System.arraycopy(values, 0, this.values, 0, this.values.length);
    }

    public int getSize() {
        return size;
    }

    @Override
    public String toString() {
        return "Layer{" +
                "dim=" + size +
                ", values=" + Arrays.toString(values) +
                '}';
    }
}
