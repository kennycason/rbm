package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class Sigmoid implements DoubleFunction {

    @Override
    public double apply(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public String toString() {
        return "sigmoid(x) = 1 / (1 + e^(-x))";
    }

}
