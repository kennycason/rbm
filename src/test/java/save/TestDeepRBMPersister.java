package save;

import math.Matrix;
import nn.rbm.RBM;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.save.DeepRBMPersister;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/22/14.
 */
public class TestDeepRBMPersister {

    private static final double DELTA = 0.0;

    @Test
    public void saveLoadTest() {
        final LayerParameters[] layerParameters = new LayerParameters[] {
            new LayerParameters().setNumRBMS(2).setVisibleUnitsPerRBM(3).setHiddenUnitsPerRBM(2),
            new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(6).setHiddenUnitsPerRBM(2),
        };

        final DeepRBMPersister deepRBMPersister = new DeepRBMPersister();
        final DeepRBM deepRBM = new DeepRBM(layerParameters, new RandomRBMFactory());

        deepRBMPersister.save(deepRBM, "/tmp/deep_rbm.csv");
        final DeepRBM deepRBM2 = deepRBMPersister.load("/tmp/deep_rbm.csv");

        for(int l = 0; l < layerParameters.length; l++) {
            for(int r = 0; r < layerParameters[l].getNumRBMS(); r++) {
                final RBM rbm = deepRBM.getRbmLayers()[l].getRBM(r);
                final RBM rbm2 = deepRBM2.getRbmLayers()[l].getRBM(r);

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
    }

}
