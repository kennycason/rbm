package nn.rbm.factory;

import math.matrix.Matrix;
import nn.rbm.Layer;
import nn.rbm.RBM;
import org.apache.log4j.Logger;

import java.util.Random;

/**
 * Created by kenny on 5/12/14.
 */
public class RandomRBMFactory implements RBMFactory {

    private static final Logger LOGGER = Logger.getLogger(RandomRBMFactory.class);

    private static final Random RANDOM = new Random();

    public RandomRBMFactory() {}

    @Override
    public RBM build(final int numVisibleNodes, final int numHiddenNodes) {
        final Layer visible = new Layer(numVisibleNodes);
        final Layer hidden = new Layer(numHiddenNodes);
        final RBM rbm = new RBM(visible, hidden);

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
