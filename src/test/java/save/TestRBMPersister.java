package save;

import math.Matrix;
import nn.rbm.RBM;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.save.RBMPersister;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/22/14.
 */
public class TestRBMPersister {

    private static final double DELTA = 0.0;

    @Test
    public void saveLoadTest() {
        final RBMPersister rbmPersister = new RBMPersister();
        final RBM rbm = new RandomRBMFactory().build(5, 2);

        rbmPersister.save(rbm, "/tmp/rbm.csv");
        final RBM rbm2 = rbmPersister.load("/tmp/rbm.csv");

        final Matrix rbmWeights = rbm.getWeights();
        final Matrix rbm2Weights = rbm2.getWeights();

        assertEquals(rbmWeights.rows(), rbm2Weights.rows());
        assertEquals(rbmWeights.columns(), rbm2Weights.columns());

        for(int i = 0; i < rbmWeights.rows(); i++) {
            for(int j = 0; j < rbmWeights.columns(); j++) {
                assertEquals(rbmWeights.get(i, j), rbm2Weights.get(i, j), DELTA);
            }
        }
    }

}
