package math;

/**
 * Created by kenny on 5/13/14.
 */
public class Vector {

    private final double[] values;

    public Vector(int size) {
        this.values = new double[size];
    }

    public Vector(double... values) {
        this.values = values;
    }

    public void set(int i, double value) {
        this.values[i] = value;
    }

    public double get(int i) {
        return this.values[i];
    }

    public double[] data() {
        return this.values;
    }

    public Vector copy() {
        final double[] copy = new double[values.length];
        System.arraycopy(this.values, 0, copy, 0, copy.length);
        return new Vector(copy);
    }

    public double dot(Vector v) {
        return dot(this.values, v.values);
    }

    public double getSquaredError(Vector v) {
        return getSquaredError(this.values, v.values);
    }

    public int size() {
        return values.length;
    }

    public static double dot(double[] a, double[] b) {
        if(a.length != b.length) { throw new IllegalArgumentException("Vector lengths must be the same!"); }
        double dot = 0.0;
        for(int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    public static double getSquaredError(double[] a, double[] b) {
        double squaredError = 0.0;
        for (int i = 0; i < a.length; i++) {
            squaredError += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return squaredError;
    }

}
