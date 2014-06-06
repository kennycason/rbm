package math.functions.distance;

import math.Matrix;

/**
 * Created by kenny on 2/16/14.
 */
public class EuclideanDistanceFunction implements DistanceFunction {

    @Override
    public double distance(Matrix item1, Matrix item2) {
        double sumSq = 0;
        for(int i = 0; i < item1.columns(); i++) {
            sumSq += Math.pow(item1.get(0, i) - item2.get(0, i), 2);
        }
        return Math.sqrt(sumSq);
    }

    @Override
    public String toString() {
        return "EuclideanDistanceFunction";
    }

}
