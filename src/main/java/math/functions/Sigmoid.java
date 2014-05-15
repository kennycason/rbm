package math.functions;

import com.google.common.base.Function;

/**
 * Created by kenny on 5/12/14.
 */
public class Sigmoid implements Function<Double, Double> {

    @Override
    public Double apply(Double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public String toString() {
        return "sigmoid(x) = 1 / (1 + e^(-x))";
    }

}
