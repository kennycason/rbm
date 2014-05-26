package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class Power implements DoubleFunction {

    private final double power;

    public Power(final double power) {
        this.power = power;
    }

    @Override
    public double apply(double v) {
        return Math.pow(v, power);
    }
}
