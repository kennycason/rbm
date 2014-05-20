package nn.rbm;

import data.image.Image;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.encode.Matrix24BitImageEncoder;
import data.image.encode.MatrixGrayscaleImageEncoder;
import data.mnist.MNISTImageLoader;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.learn.ContrastiveDivergenceRBM;
import nn.rbm.learn.DeepContrastiveDivergenceRBM;
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

    private static final RandomRBMFactory RBM_FACTORY = new RandomRBMFactory();
    @Test
    public void train() {
        final RBM rbm = RBM_FACTORY.build(6, 3);
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
        final RBM rbm = RBM_FACTORY.build(6, 4);
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

        final RBM rbm = RBM_FACTORY.build(imageDim, 200);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(5000));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(dataSet.row(0), 28, 0.5));
        cdRBM.learn(dataSet);

        for(int i = 0; i < dataSet.rows(); i++) {
            final Matrix testData = new ImmutableMatrix(new double[][] {dataSet.row(i)});
            final Matrix hidden = cdRBM.runVisible(testData);
            final Matrix visual = cdRBM.runHidden(hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }

    }

    @Test
    public void fewNumbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 25);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(15000));

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
            final Matrix testData = new ImmutableMatrix(new double[][] {trainingSet.row(i)});
            final Matrix hidden = cdRBM.runVisible(testData);
            final Matrix visual = cdRBM.runHidden(hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }

    }

    @Test
    public void number() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 20);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(15000));
        final Matrix trainingData = new ImmutableMatrix(new double[][] { totalDataSet.row(0) });

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0), 28, 0.5));

        cdRBM.learn(trainingData);

        final Matrix hidden = cdRBM.runVisible(trainingData);
        final Matrix visual = cdRBM.runHidden(hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
    }

    @Test
    public void imageMicro() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(10000));

        cdRBM.learn(jetMatrix);

        final Matrix hidden = cdRBM.runVisible(jetMatrix);
        final Matrix visual = cdRBM.runHidden(hidden);
        final Image outImage = new Matrix24BitImageDecoder(19).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered.jpg");
    }


    @Test
    public void imageSmall() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(5000));

        cdRBM.learn(jetMatrix);

        final Matrix hidden = cdRBM.runVisible(jetMatrix);
        final Matrix visual = cdRBM.runHidden(hidden);
        final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered.jpg");
    }

    @Test
    public void imageGrayScale() {
        final Image jetImage = new Image("/data/fighter_jet_micro.jpg");
        final Matrix jetMatrix = new MatrixGrayscaleImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergenceRBM cdRBM = new ContrastiveDivergenceRBM(rbm, new LearningParameters().setEpochs(10000));

        cdRBM.learn(jetMatrix);

        final Matrix hidden = cdRBM.runVisible(jetMatrix);
        final Matrix visual = cdRBM.runHidden(hidden);
        final Image outImage = new data.image.decode.MatrixGrayscaleImageDecoder(19).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered.jpg");
    }

    @Test
    public void deepRBM() {
        // 28 * 28 input (784)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(16).setVisibleUnitsPerRBM(49).setHiddenUnitsPerRBM(10),    // 784 in, 160 out
                new LayerParameters().setNumRBMS(8).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 160 in, 80 out
                new LayerParameters().setNumRBMS(4).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 80 in, 40 out
                new LayerParameters().setNumRBMS(2).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 40 in, 20 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(100),    // 20 in, 100 out
        };

        DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);
        // LOGGER.info(deepRBM);


        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final DeepContrastiveDivergenceRBM cdRBM = new DeepContrastiveDivergenceRBM(deepRBM, new LearningParameters().setEpochs(1000));
        final Matrix trainingData = new ImmutableMatrix(new double[][] { totalDataSet.row(0) });

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0), 28, 0.5));

        cdRBM.learn(trainingData);

        final Matrix hidden = cdRBM.runVisible(trainingData);
        final Matrix visual = cdRBM.runHidden(hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
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
