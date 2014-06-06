package math.functions.distance;

import math.Matrix;

/**
 * Created by kenny on 2/16/14.
 */
public class DiscreteDistanceFunction implements DistanceFunction {

    private static final double DELTA = 0.05;

    @Override
    public double distance(Matrix item1, Matrix item2) {
        for(int i = 0; i < item1.columns(); i++) {
            if(Math.abs(item1.get(0, i) - item2.get(0, i)) > DELTA) {
                return 1.0;
            }
        }
        return 0.0;
    }

    @Override
    public String toString() {
        return "DiscreteDistanceFunction";
    }

}
