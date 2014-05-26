package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

/**
 * Created by kenny on 5/24/14.
 */
public class OneMinusX implements DoubleFunction {
    @Override
    public double apply(double x) {
        return 1 - x;
    }
}
