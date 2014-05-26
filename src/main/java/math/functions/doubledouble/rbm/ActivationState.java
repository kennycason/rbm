package math.functions.doubledouble.rbm;

import cern.colt.function.tdouble.DoubleDoubleFunction;

/**
 * Created by kenny on 5/25/14.
 */
public class ActivationState implements DoubleDoubleFunction {
    @Override
    public double apply(double x, double y) {
        return x >= y ? 1.0 : 0.0;
    }
}
