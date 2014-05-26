package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class RangedSigmoid implements DoubleFunction {
    private final double min;

    private final double max;

    public RangedSigmoid(final double min, final double max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public double apply(double x) {
        return min + ((max - min) / (1.0 + Math.exp(-x)));
    }

    @Override
    public String toString() {
        return "sigmoid(x) = min + ((max - min) / (1 + e^(-x)))";
    }
}
