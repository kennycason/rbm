package nn.rbm;

import data.mnist.MNISTImageLoader;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import nn.rbm.learn.ContrastiveDivergenceRBM;
import nn.rbm.learn.LearningParameters;
import org.apache.log4j.Logger;
import org.junit.Ignore;
import org.junit.Test;
import utils.PrettyPrint;

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
        final Matrix testData = new ImmutableMatrix(new double[][] {{0,0,0,1,1,0}, {0,0,1,1,0,0}});
        final Matrix hidden = cdRBM.runVisible(testData);
        LOGGER.info(hidden);
        final Matrix visual = cdRBM.runHidden(hidden);
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

    @Ignore
    public void numbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix dataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = dataSet.dim() / dataSet.rows(); // 784

        final RBM rbm = RBMFactory.buildRandomRBM(imageDim, 200);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(5000));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(dataSet.row(0), 28, 0.5));
        cdRBM.learn(dataSet);

        for(int i = 0; i < dataSet.rows(); i++) {
            LOGGER.info("Data Index: " + i);
            final Matrix testData = new ImmutableMatrix(new double[][] {dataSet.row(i)});
            final Matrix hidden = cdRBM.runVisible(testData);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(hidden.row(0), 28, 0.5));
            final Matrix visual = cdRBM.runHidden(hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }

    }

    @Test
    public void fewNumbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBMFactory.buildRandomRBM(imageDim, 15);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(5000));

        final Matrix trainingSet = new ImmutableMatrix(new double[][] {
                totalDataSet.row(0),
                totalDataSet.row(100),
                totalDataSet.row(200),
                totalDataSet.row(300),
                totalDataSet.row(400),
                totalDataSet.row(500),
                totalDataSet.row(600)
        });

        for(int i = 0; i < trainingSet.rows(); i++) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingSet.row(i), 28, 0.5));
        }

        cdRBM.learn(trainingSet);

        for(int i = 0; i < trainingSet.rows(); i++) {
            LOGGER.info("Data Index: " + i);
            final Matrix testData = new ImmutableMatrix(new double[][] {trainingSet.row(i)});
            final Matrix hidden = cdRBM.runVisible(testData);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(hidden.row(0), 28, 0.5));
            final Matrix visual = cdRBM.runHidden(hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }

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
