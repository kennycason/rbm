package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

import java.util.Random;

/**
 * Created by kenny on 5/24/14.
 */
public class RandomDouble implements DoubleFunction {

    private static final Random RANDOM = new Random();

    private final double scalar;

    public RandomDouble() {
        this(1.0);
    }

    public RandomDouble(final double scalar) {
        this.scalar = scalar;
    }

    @Override
    public double apply(double v) {
        return RANDOM.nextDouble() * scalar;
    }

}
