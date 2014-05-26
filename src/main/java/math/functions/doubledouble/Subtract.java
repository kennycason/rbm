package math.functions.doubledouble;

import cern.colt.function.tdouble.DoubleDoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class Subtract implements DoubleDoubleFunction {
    @Override
    public double apply(double v, double v2) {
        return v - v2;
    }
}
