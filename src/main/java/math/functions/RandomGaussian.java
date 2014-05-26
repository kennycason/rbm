package math.functions;

import cern.colt.function.tdouble.DoubleFunction;

import java.util.Random;

/**
 * Created by kenny on 5/24/14.
 */
public class RandomGaussian implements DoubleFunction {

    private static final Random RANDOM = new Random();

    @Override
    public double apply(double v) {
        return RANDOM.nextGaussian() * 0.1;
    }

}
