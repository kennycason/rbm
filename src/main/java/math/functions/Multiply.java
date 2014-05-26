package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class Multiply implements DoubleFunction {

    private final double value;

    public Multiply(double value) {
        this.value = value;
    }

    @Override
    public double apply(double v) {
        return v * value;
    }
}
