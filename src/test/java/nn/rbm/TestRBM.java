package nn.rbm;

import cern.colt.function.tdouble.DoubleFunction;
import data.image.Image;
import data.image.decode.Matrix1BitImageDecoder;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.decode.Matrix8BitImageDecoder;
import data.image.decode.MatrixGrayscaleImageDecoder;
import data.image.encode.Matrix24BitImageEncoder;
import data.image.encode.Matrix8BitImageEncoder;
import data.image.encode.MatrixGrayscaleImageEncoder;
import data.mnist.MNISTImageLoader;
import math.DenseMatrix;
import math.Matrix;
import math.functions.Round;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.learn.ContrastiveDivergence;
import nn.rbm.learn.DeepContrastiveDivergence;
import nn.rbm.learn.LearningParameters;
import nn.rbm.learn.MultiThreadedDeepContrastiveDivergence;
import nn.rbm.learn.RecurrentContrastiveDivergence;
import org.apache.log4j.Logger;
import org.junit.Ignore;
import org.junit.Test;
import utils.PrettyPrint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

        contrastiveDivergence.learn(rbm, buildBetterSampleTrainingData());

        // fetch two recommendations
        final Matrix testData = DenseMatrix.make(new double[][]{{0, 0, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0}});
        final Matrix hidden = contrastiveDivergence.runVisible(rbm, testData);
        LOGGER.info(testData);
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

    private Matrix buildBetterSampleTrainingData() {
        return DenseMatrix.make(
                new double[][] {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0}}
        );
    }

    @Ignore
    public void numbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix dataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = dataSet.dim() / dataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 200);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(5000));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(dataSet.row(0).toArray(), 28, 0.5));
        contrastiveDivergence.learn(rbm, dataSet);

        for(int i = 0; i < dataSet.rows(); i++) {
            final Matrix testData = DenseMatrix.make(dataSet.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, testData);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
        }

    }

    @Test
    public void fewNumbersAsSingleMAtrix() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 25);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(10000));

        final Matrix trainingSet = DenseMatrix.make(new double[][] {
                totalDataSet.row(0).toArray(),
                totalDataSet.row(100).toArray(),
                totalDataSet.row(200).toArray(),
                totalDataSet.row(300).toArray(),
                totalDataSet.row(400).toArray(),
                totalDataSet.row(500).toArray(),
                totalDataSet.row(600).toArray()
        });

        for(int i = 0; i < trainingSet.rows(); i++) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingSet.row(i).toArray(), 28, 0.5));
        }

        contrastiveDivergence.learn(rbm, trainingSet);

        for(int i = 0; i < trainingSet.rows(); i++) {
            final Matrix testData = DenseMatrix.make(trainingSet.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, testData);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
        }

    }


    @Test
    public void fewNumbersAsListOfMatrices() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 25);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(10000));

        final List<Matrix> trainingSet = Arrays.asList(
                DenseMatrix.make(totalDataSet.row(0)),
                DenseMatrix.make(totalDataSet.row(100)),
                DenseMatrix.make(totalDataSet.row(200)),
                DenseMatrix.make(totalDataSet.row(300)),
                DenseMatrix.make(totalDataSet.row(400)),
                DenseMatrix.make(totalDataSet.row(500)),
                DenseMatrix.make(totalDataSet.row(600))
        );

        for(Matrix data : trainingSet) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(data.row(0).toArray(), 28, 0.5));
        }

        contrastiveDivergence.learn(rbm, trainingSet);

        for(Matrix data : trainingSet) {
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, data);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
        }

    }

    @Test
    public void number() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 20);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(15000));
        final Matrix trainingData = DenseMatrix.make(totalDataSet.row(0));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0).toArray(), 28, 0.5));

        contrastiveDivergence.learn(rbm, trainingData);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, trainingData);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
    }

    @Test
    public void imageMicro() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.columns(), 50);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(1000));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new Matrix24BitImageDecoder(19).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered.bmp");
    }


    @Test
    public void imageSmall24Bit() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.columns(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(5000));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_24bit.bmp");
    }

    @Test
    public void imageSmall8Bit() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix8BitImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.columns(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(1000));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new Matrix8BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_8bit.bmp");
    }

    @Test
    public void imageGrayScale() {
        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new MatrixGrayscaleImageEncoder().encode(jetImage);

        final RBM rbm = RBM_FACTORY.build(jetMatrix.columns(), 100);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(250));

        contrastiveDivergence.learn(rbm, jetMatrix);

        final Matrix hidden = contrastiveDivergence.runVisible(rbm, jetMatrix);
        final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
        final Image outImage = new MatrixGrayscaleImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_grayscale.bmp");
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
        final Matrix trainingData = DenseMatrix.make(totalDataSet.row(0));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0).toArray(), 28, 0.5));

        contrastiveDivergence.learn(deepRBM, trainingData);

        final Matrix hidden = contrastiveDivergence.runVisible(deepRBM, trainingData);

        final Matrix visual = contrastiveDivergence.runHidden(deepRBM, hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
    }

    @Test
    public void deepRBM60KNumbers() {
        // 28 * 28 input (784)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(392).setVisibleUnitsPerRBM(2).setHiddenUnitsPerRBM(10),    // 784 in, 3920 out
                new LayerParameters().setNumRBMS(196).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),    // 3920 in, 1960 out
                new LayerParameters().setNumRBMS(98).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 1960 in, 980 out
                new LayerParameters().setNumRBMS(70).setVisibleUnitsPerRBM(14).setHiddenUnitsPerRBM(4),     // 980 in, 280 out
                new LayerParameters().setNumRBMS(20).setVisibleUnitsPerRBM(14).setHiddenUnitsPerRBM(3),    // 280 in, 60 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(60).setHiddenUnitsPerRBM(10)     // 60 in, 10 out
//                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(6).setHiddenUnitsPerRBM(2),    // 60 in, 20 out
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(100),    // 20 in, 100 out
        };

        DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);

        LOGGER.info("Loading training & test data");
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix trainingData = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);  // 60,000 inputs
        final Matrix testData = mnistImageLoader.loadIdx3("/data/t10k-images-idx3-ubyte").divide(255.0);  // 10,000 inputs
        LOGGER.info("Finished loading");

        final MultiThreadedDeepContrastiveDivergence contrastiveDivergence =
                new MultiThreadedDeepContrastiveDivergence(new LearningParameters().setEpochs(450));

        //contrastiveDivergence.learn(deepRBM, DenseMatrix.make(trainingData));
        contrastiveDivergence.learn(deepRBM, DenseMatrix.make(testData));

        for(int i = 0; i < testData.rows(); i++) {
            final Matrix dataMatrix = DenseMatrix.make(testData.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(deepRBM, dataMatrix);
            final Matrix visual = contrastiveDivergence.runHidden(deepRBM, hidden);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
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
        outImage.save("/tmp/fighter_rendered_small_24bit_deep.bmp");
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
        outImage.save("/tmp/fighter_rendered_large_24bit_deep.bmp");
    }

    @Test
    public void imageSmall24BitDeepRBMTestFeatures() {
        // 100 * 63 * 24 input (151200)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(200).setVisibleUnitsPerRBM(756).setHiddenUnitsPerRBM(100),    // 151,200 in, 20,000 out
                new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 20,000 in, 5,000 out
                new LayerParameters().setNumRBMS(50).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 5,000 in, 500 out
                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 500 in, 250 out
                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 250 in, 50 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(100),    // 50 in, 100 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),    // 10 in, 10 out
        };

        DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);
        final DeepContrastiveDivergence deepContrastiveDivergence = new DeepContrastiveDivergence(new LearningParameters().setEpochs(100));

        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        deepContrastiveDivergence.learn(deepRBM, jetMatrix);

        final List<Matrix> features = new ArrayList<>(10);
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{1,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,1,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,1,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,1,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,1, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 1,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,1,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,1,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,1,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,1}}));
        features.add(DenseMatrix.make(new double[][] {{1,1,1,1,1, 1,1,1,1,1}}));

        int i = 0;
        for(Matrix feature : features) {
            final Matrix visual = deepContrastiveDivergence.runHidden(deepRBM, feature);
            final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
            outImage.save("/tmp/fighter_rendered_small_24bit_deep_feature_" + i + ".bmp");
            i++;
        }
    }


    @Test
    public void numbersTestFeatures() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim, 7);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(new LearningParameters().setEpochs(15000));

        final Matrix trainingSet = DenseMatrix.make(new double[][] {
                totalDataSet.row(0).toArray(),
                totalDataSet.row(100).toArray(),
                totalDataSet.row(200).toArray(),
                totalDataSet.row(300).toArray(),
                totalDataSet.row(400).toArray(),
                totalDataSet.row(500).toArray(),
                totalDataSet.row(600).toArray()
        });

        for(int i = 0; i < trainingSet.rows(); i++) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingSet.row(i).toArray(), 28, 0.5));
        }

        contrastiveDivergence.learn(rbm, trainingSet);

        // activate single features
        final List<Matrix> features = new ArrayList<>(7);
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0}}));
        features.add(DenseMatrix.make(new double[][] {{1,0,0,0,0, 0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,1,0,0,0, 0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,1,0,0, 0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,1,0, 0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,1, 0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 1,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,1}}));
        features.add(DenseMatrix.make(new double[][] {{1,1,1,1,1, 1,1}}));
        int i = 0;
        for(Matrix feature : features) {
            final Matrix visual = contrastiveDivergence.runHidden(rbm, feature);
            final Image outImage = new Matrix1BitImageDecoder(28).decode(visual);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
            outImage.save("/tmp/numbers_rendered_feature_" + i + ".bmp");
            i++;
        }

        // actual rbm reconstructions
        for(int j = 0; j < trainingSet.rows(); j++) {
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, DenseMatrix.make(trainingSet.row(j)));
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            final Image outImage = new Matrix1BitImageDecoder(28).decode(visual);
            LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
            outImage.save("/tmp/numbers_rendered_" + j + ".bmp");
            i++;
        }
    }

    @Test
    public void recurrentNumber() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim * 2, 20);  // two times the input because of the recurrent input
        final RecurrentContrastiveDivergence recurrentContrastiveDivergence = new RecurrentContrastiveDivergence(new LearningParameters().setEpochs(1000));
        final Matrix trainingData = DenseMatrix.make(totalDataSet.row(0));

        LOGGER.info("\n" + PrettyPrint.toPixelBox(trainingData.row(0).toArray(), 28, 0.5));

        recurrentContrastiveDivergence.learn(rbm, Arrays.asList(trainingData));

        final Matrix hidden = recurrentContrastiveDivergence.runVisible(rbm, trainingData);
        final Matrix visual = recurrentContrastiveDivergence.runHidden(rbm, hidden);
        LOGGER.info("\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
    }


    @Test
    public void recurrentNumbers() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final RBM rbm = RBM_FACTORY.build(imageDim * 2, 30); // two times the input because of the recurrent input
        final RecurrentContrastiveDivergence recurrentContrastiveDivergence = new RecurrentContrastiveDivergence(new LearningParameters().setEpochs(7500).setLearningRate(0.1));

        final List<Matrix> trainingData = new ArrayList<>(10);
        trainingData.add(DenseMatrix.make(totalDataSet.row(0)));      // 5
        trainingData.add(DenseMatrix.make(totalDataSet.row(1)));      // 0
        trainingData.add(DenseMatrix.make(totalDataSet.row(2)));      // 4
        trainingData.add(DenseMatrix.make(totalDataSet.row(3)));      // 1
        trainingData.add(DenseMatrix.make(totalDataSet.row(4)));      // 9
        trainingData.add(DenseMatrix.make(totalDataSet.row(5)));     // 2
     //   trainingData.add(DenseMatrix.make(totalDataSet.row(6)));     // 1
        trainingData.add(DenseMatrix.make(totalDataSet.row(7)));    // 3
    //    trainingData.add(DenseMatrix.make(totalDataSet.row(8)));   // 1
        trainingData.add(DenseMatrix.make(totalDataSet.row(9)));     // 4


        for(Matrix data : trainingData) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(data.row(0).toArray(), 28, 0.5));
        }

        recurrentContrastiveDivergence.learn(rbm, trainingData);

        // see if network consecutively draws numbers
        final DoubleFunction round = new Round(0.6);
        Matrix hidden;
        Matrix visual = trainingData.get(trainingData.size() - 1);
        LOGGER.info("Input : " + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
        for(int i = 0; i < trainingData.size() - 1; i++) {
            hidden = recurrentContrastiveDivergence.runVisible(rbm, visual);
            visual = recurrentContrastiveDivergence.runHidden(rbm, hidden);

            visual = DenseMatrix.make(visual.data().viewPart(0, imageDim, 1, imageDim)); // trim off the previous input and only pass on the prediction
            LOGGER.info("Guess of what comes next\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.6));
            visual.apply(round);

        }
    }


    /**
     * learn sequence 4 bit binary number
     *
     * doesn't work so well heh
     */
    @Test
    public void recurrentBinary() {

        final RBM rbm = RBM_FACTORY.build(8, 30); // two times the input because of the recurrent input
        final RecurrentContrastiveDivergence recurrentContrastiveDivergence = new RecurrentContrastiveDivergence(new LearningParameters().setEpochs(10000));

        final List<Matrix> trainingData = new ArrayList<>(16);
        trainingData.add(DenseMatrix.make(new double[][]{{0,0,0,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,0,0,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,0,1,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,0,1,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,1,0,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,1,0,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,1,1,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{0,1,1,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,0,0,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,0,0,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,0,1,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,0,1,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,1,0,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,1,0,1}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,1,1,0}}));
        trainingData.add(DenseMatrix.make(new double[][] {{1,1,1,1}}));

        for(Matrix data : trainingData) {
            LOGGER.info(PrettyPrint.toPixelBox(data.toArray(), 0.5));
        }

        recurrentContrastiveDivergence.learn(rbm, trainingData);

        // see if network consecutively draws numbers
        final DoubleFunction round = new Round(0.5);
        Matrix hidden;
        Matrix visual = trainingData.get(trainingData.size() - 1);
        LOGGER.info("Input : " + PrettyPrint.toPixelBox(visual.toArray(), 0.5));
        for(int i = 0; i < trainingData.size() - 1; i++) {
            hidden = recurrentContrastiveDivergence.runVisible(rbm, visual);
            visual = recurrentContrastiveDivergence.runHidden(rbm, hidden);

            visual = DenseMatrix.make(visual.data().viewPart(0, 4, 1, 4)); // trim off the previous input and only pass on the prediction
            LOGGER.info("Guess of what came before: " + PrettyPrint.toPixelBox(visual.toArray(), 0.5));
            visual.apply(round);
        }
    }

}
