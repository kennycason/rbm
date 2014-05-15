package math;

/**
 * Created by kenny on 5/13/14.
 */
public class Vector {

    private Vector() {}

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

    public static double getMeanSquaredError(double[][] as, double[][] bs) {
        double error = 0.0;
        for (int i = 0; i < as.length; i++) {
            error += getSquaredError(as[i], bs[i]);
        }
        return error / as.length;
    }

}
