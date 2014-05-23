package math.functions;

import com.google.common.base.Function;

/**
 * Created by kenny on 5/23/14.
 */
public class OneMinusX implements Function<Double, Double> {
    @Override
    public Double apply(Double x) {
        return 1 - x;
    }
}
