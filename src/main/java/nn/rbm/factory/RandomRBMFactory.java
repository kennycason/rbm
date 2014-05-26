package nn.rbm.factory;

import math.Matrix;
import nn.rbm.RBM;

import java.util.Random;

/**
 * Created by kenny on 5/12/14.
 */
public class RandomRBMFactory implements RBMFactory {

    private static final Random RANDOM = new Random();

    public RandomRBMFactory() {}

    @Override
    public RBM build(final int numVisibleNodes, final int numHiddenNodes) {
        final RBM rbm = new RBM(numVisibleNodes, numHiddenNodes);

        final Matrix weights = rbm.getWeights();
        for(int i = 0; i < numVisibleNodes; i++) {
            for(int j = 0; j < numHiddenNodes; j++) {
                weights.set(i, j, randomWeight());
            }
        }
        return rbm;
    }

    private static double randomWeight() {
        return RANDOM.nextGaussian() * 0.1;
    }

}
