package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class Round implements DoubleFunction {
    private final double threshold;

    public Round() {
        this(0.80);
    }

    public Round(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public double apply(double x) {
        return x >= threshold ? 1.0 : 0.0;
    }

    @Override
    public String toString() {
        return "Round{" +
                "threshold=" + threshold +
                '}';
    }
}
