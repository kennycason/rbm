package math.functions;

import com.google.common.base.Function;

/**
 * Created by kenny on 5/24/14.
 */
public class RangedSigmoid implements Function<Double, Double> {

    private final double min;

    private final double max;

    public RangedSigmoid(final double min, final double max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public Double apply(Double x) {
        return min + ((max - min) / (1.0 + Math.exp(-x)));
    }

    @Override
    public String toString() {
        return "sigmoid(x) = min + ((max - min) / (1 + e^(-x)))";
    }

}
