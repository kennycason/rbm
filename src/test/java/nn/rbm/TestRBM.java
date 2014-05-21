package nn.rbm;

import data.image.Image;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.decode.Matrix8BitImageDecoder;
import data.image.decode.MatrixGrayscaleImageDecoder;
import data.image.encode.Matrix24BitImageEncoder;
import data.image.encode.Matrix8BitImageEncoder;
import data.image.encode.MatrixGrayscaleImageEncoder;
import data.mnist.MNISTImageLoader;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.learn.ContrastiveDivergence;
import nn.rbm.learn.DeepContrastiveDivergence;
import nn.rbm.learn.LearningParameters;
import nn.rbm.learn.RecurrentContrastiveDivergence;
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
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(25000));
        LOGGER.info(rbm);

        contrastiveDivergence.learn(rbm, buildBetterSampleTrainingData());
        LOGGER.info(rbm);

        // fetch two recommendations
        final Matrix testData = new ImmutableMatrix(new double[][] {{0,0,0,1,1,0}, {0,0,1,1,0,0}});
        final Matrix hidden = contrastiveDivergence.runVisible(rbm, testData);
        LOGGER.info(hidden);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        LOGGER.info(visual);
    }

    @Test
    public void daydream() {
        final RBM rbm = RBM_FACTORY.build(6, 4);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(25000));

        contrastiveDivergence.learn(rbm, buildBetterSampleTrainingData());

        Set<Matrix> visualizations = contrastiveDivergence.dayDream(rbm, buildBetterSampleTrainingData(), 10);
        LOGGER.info(visualizations);
    }

    @Ignore
    public void numbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix dataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = dataSet.dim() / dataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 200);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(5000));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(dataSet.row(0), 28, 0.5));
        contrastiveDivergence.learn(rbm, dataSet);

        for(int i = 0; i < dataSet.rows(); i++) {
            final Matrix testData = new ImmutableMatrix(new double[][] {dataSet.row(i)});
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, testData);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }

    }

    @Test
    public void fewNumbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 25);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(15000));

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

        contrastiveDivergence.learn(rbm, trainingSet);

        for(int i = 0; i < trainingSet.rows(); i++) {
            final Matrix testData = new ImmutableMatrix(trainingSet.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, testData);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }

    }

    @Test
    public void number() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 20);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(15000));
        final Matrix trainingData = new ImmutableMatrix(totalDataSet.row(0));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0), 28, 0.5));

        contrastiveDivergence.learn(rbm, trainingData);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, trainingData);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
    }

    @Test
    public void imageMicro() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(10000));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new Matrix24BitImageDecoder(19).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered.jpg");
    }


    @Test
    public void imageSmall24Bit() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(5000));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_24bit.jpg");
    }

    @Test
    public void imageSmall8Bit() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix8BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(1000));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new Matrix8BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_8bit.jpg");
    }

    @Test
    public void imageGrayScale() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new MatrixGrayscaleImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.cols(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(250));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new MatrixGrayscaleImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_grayscale.jpg");
    }

    @Test
    public void deepRBMSingleNumber() {
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

        final DeepContrastiveDivergence contrastiveDivergence = new DeepContrastiveDivergence(new LearningParameters().setEpochs(500));
        final Matrix trainingData = new ImmutableMatrix(totalDataSet.row(0));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0), 28, 0.5));

        contrastiveDivergence.learn(deepRBM, trainingData);

        final Matrix hidden = contrastiveDivergence.runVisible(deepRBM, trainingData);

        final Matrix visual = contrastiveDivergence.runHidden(deepRBM, hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
    }

    @Test
    public void deepRBM60KNumbers() {
        // 28 * 28 input (784)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(392).setVisibleUnitsPerRBM(2).setHiddenUnitsPerRBM(10),    // 784 in, 3920 out
                new LayerParameters().setNumRBMS(196).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),    // 3920 in, 1960 out
                new LayerParameters().setNumRBMS(88).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 1960 in, 880 out
                new LayerParameters().setNumRBMS(44).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 880 in, 440 out
                new LayerParameters().setNumRBMS(22).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),    // 440 in, 220 out
                new LayerParameters().setNumRBMS(11).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),    // 220 in, 110 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(110).setHiddenUnitsPerRBM(100),    // 110 in, 100 out
        };

        DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);

        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix trainingData = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);  // 60,000 inputs
        final Matrix testData = mnistImageLoader.loadIdx3("/data/t10k-images-idx3-ubyte").divide(255.0);  // 10,000 inputs

        final DeepContrastiveDivergence contrastiveDivergence = new DeepContrastiveDivergence(new LearningParameters().setEpochs(1));

        contrastiveDivergence.learn(deepRBM, trainingData);

        for(double[] data : testData.data()) {
            final Matrix dataMatrix = new ImmutableMatrix(data);
            final Matrix hidden = contrastiveDivergence.runVisible(deepRBM, dataMatrix);
            final Matrix visual = contrastiveDivergence.runHidden(deepRBM, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
        }
    }

    @Test
    public void imageSmall24BitDeepRBM() {
        // 100 * 63 * 24 input (151200)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(200).setVisibleUnitsPerRBM(756).setHiddenUnitsPerRBM(100),    // 151,200 in, 20,000 out
                new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 20,000 in, 5,000 out
                new LayerParameters().setNumRBMS(50).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 5,000 in, 500 out
                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 500 in, 250 out
                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 250 in, 50 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(100),    // 50 in, 100 out
        };

        DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);
        final DeepContrastiveDivergence deepContrastiveDivergence = new DeepContrastiveDivergence(new LearningParameters().setEpochs(100));

        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        deepContrastiveDivergence.learn(deepRBM, jetMatrix);

        final Matrix hidden = deepContrastiveDivergence.runVisible(deepRBM, jetMatrix);
        final Matrix visual = deepContrastiveDivergence.runHidden(deepRBM, hidden);
        final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_24bit_deep.jpg");
    }


    @Test
    public void imageLarge24BitDeepRBM() {
        // 400 * 250 * 24 input (2,400,000)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(5000).setVisibleUnitsPerRBM(480).setHiddenUnitsPerRBM(100),    // 2,400,000 in, 500,000 out
                new LayerParameters().setNumRBMS(2500).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 500,000 in, 125,000 out
                new LayerParameters().setNumRBMS(1250).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 125,000 in, 12,500 out
                new LayerParameters().setNumRBMS(625).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 12,500 in, 6,250 out
                new LayerParameters().setNumRBMS(250).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 6,250 in, 1,250 out
                new LayerParameters().setNumRBMS(125).setVisibleUnitsPerRBM(10).setHiddenUnitsPerRBM(5),    // 1,250 in, 625 out
                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 625 in, 125 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(125).setHiddenUnitsPerRBM(100),    // 125 in, 100 out
        };

        DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);
        final DeepContrastiveDivergence deepContrastiveDivergence = new DeepContrastiveDivergence(new LearningParameters().setEpochs(11));

        final Image jetImage = new Image("/data/fighter_jet.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        deepContrastiveDivergence.learn(deepRBM, jetMatrix);

        final Matrix hidden = deepContrastiveDivergence.runVisible(deepRBM, jetMatrix);
        final Matrix visual = deepContrastiveDivergence.runHidden(deepRBM, hidden);
        final Image outImage = new Matrix24BitImageDecoder(250).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_large_24bit_deep.jpg");
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

    @Test
    public void recurrentNumber() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 20);
        final RecurrentContrastiveDivergence recurrentContrastiveDivergence = new RecurrentContrastiveDivergence(new LearningParameters().setEpochs(100));
        final Matrix trainingData = new ImmutableMatrix(totalDataSet.row(0));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0), 28, 0.5));

        recurrentContrastiveDivergence.learn(rbm, trainingData);

        final Matrix hidden = recurrentContrastiveDivergence.runVisible(rbm, trainingData);
        final Matrix visual = recurrentContrastiveDivergence.runHidden(rbm, hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0), 28, 0.5));
    }

}
