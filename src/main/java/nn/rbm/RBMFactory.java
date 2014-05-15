package nn.rbm;

import math.matrix.Matrix;
import org.apache.log4j.Logger;

import java.util.Random;

/**
 * Created by kenny on 5/12/14.
 */
public class RBMFactory {

    private static final Logger LOGGER = Logger.getLogger(RBMFactory.class);

    private static final Random RANDOM = new Random();

    private RBMFactory() {}

    public static RBM buildRandomRBM(final int numVisibleNodes, final int numHiddenNodes) {
        final Layer visible = new Layer(numVisibleNodes);
        for(int i = 0; i < numVisibleNodes; i++) {
            visible.setBias(i, randomWeight());
        }

        final Layer hidden = new Layer(numHiddenNodes);
        for(int i = 0; i < numHiddenNodes; i++) {
            hidden.setBias(i, randomWeight());
        }

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
