package nn.rbm;

import java.util.Arrays;

/**
 * Created by kenny on 5/12/14.
 */
public class Layer {

    private final int size;

    private double[] values;

    private double[] bias;

    public Layer(final int size) {
        this.size = size;
        this.bias = new double[size];
        this.values = new double[size];
    }

    public double sumBiasValues() {
        double biasSum = 0;
        for(int i = 0; i < size; i++) {
            biasSum += values[i] * bias[i];
        }
        return biasSum;
    }

    public double biasValue(int i) {
        return values[i] * bias[i];
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        System.arraycopy(values, 0, this.values, 0, this.values.length);
    }

    public void setBias(int i, double bias) {
        this.bias[i] = bias;
    }

    public double[] getBias() {
        return bias;
    }
    public int getSize() {
        return size;
    }

    @Override
    public String toString() {
        return "Layer{" +
                "dim=" + size +
                ", values=" + Arrays.toString(values) +
                ", bias=" + Arrays.toString(bias) +
                '}';
    }
}
