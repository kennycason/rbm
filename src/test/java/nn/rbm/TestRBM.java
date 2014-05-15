package nn.rbm;

import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import nn.rbm.learn.ContrastiveDivergenceRBM;
import nn.rbm.learn.LearningParameters;
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
        final RBM rbm = RBMFactory.buildRandomRBM(6, 3);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(25000));
        LOGGER.info(rbm);

        cdRBM.learn(buildBetterSampleTrainingData());
        LOGGER.info(rbm);

        // fetch two recommendations
        final Matrix dataSet = new ImmutableMatrix(new double[][] {{0,0,0,1,1,0}, {0,0,1,1,0,0}});
        Matrix hidden = cdRBM.runVisible(dataSet);
        LOGGER.info(hidden);
        Matrix visual = cdRBM.runHidden(hidden);
        LOGGER.info(visual);
    }

    @Test
    public void daydream() {
        final RBM rbm = RBMFactory.buildRandomRBM(6, 4);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(25000));

        cdRBM.learn(buildBetterSampleTrainingData());

        Set<Matrix> visualizations = cdRBM.dayDream(buildBetterSampleTrainingData(), 10);
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
