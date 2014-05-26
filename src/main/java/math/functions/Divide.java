package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class Divide implements DoubleFunction {

    private final double divisor;

    public Divide(double divisor) {
        this.divisor = divisor;
    }

    @Override
    public double apply(double v) {
        return v / divisor;
    }
}
