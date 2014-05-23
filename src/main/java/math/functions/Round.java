package math.functions;

import com.google.common.base.Function;

/**
 * Created by kenny on 5/12/14.
 */
public class Round implements Function<Double, Double> {

    private final double threshold;

    public Round() {
        this(0.80);
    }

    public Round(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public Double apply(Double x) {
        return x >= threshold ? 1.0 : 0.0;
    }

    @Override
    public String toString() {
        return "Round{" +
                "threshold=" + threshold +
                '}';
    }
}
