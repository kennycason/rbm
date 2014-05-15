package nn.rbm;

import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import org.apache.log4j.Logger;
import org.junit.Test;

import java.util.Set;

/**
 * Created by kenny on 5/12/14.
 */
public class TestRBM {

    private static final Logger LOGGER = Logger.getLogger(TestRBM.class);

    @Test
    public void train() {
        final RBM rbm = RBMFactory.buildRandomRBM(6, 4);
        rbm.setLearningRate(0.1);
        LOGGER.info(rbm);
        rbm.train(buildBetterSampleTrainingData(), 15000);
        LOGGER.info(rbm);

        // fetch two recommendations
        final Matrix dataSet = new ImmutableMatrix(new double[][] {{0,0,0,1,1,0}, {0,0,1,1,0,0}});
        Matrix hidden = rbm.runVisible(dataSet);
        LOGGER.info(hidden);
        Matrix visual = rbm.runHidden(hidden);
        LOGGER.info(visual);
    }

    @Test
    public void daydream() {
        final RBM rbm = RBMFactory.buildRandomRBM(6, 4);
        rbm.setLearningRate(0.1);
        rbm.train(buildBetterSampleTrainingData(), 15000);

        Set<Matrix> visualizations = rbm.dayDream (buildBetterSampleTrainingData(), 100);
        LOGGER.info(visualizations);
    }

    private static Matrix buildBetterSampleTrainingData() {
        return new ImmutableMatrix(
                new double[][] {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0}}
        );
    }

}
