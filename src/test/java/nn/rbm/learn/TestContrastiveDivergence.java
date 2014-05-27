package nn.rbm.learn;

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
import nn.rbm.RBM;
import nn.rbm.factory.RandomRBMFactory;
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
public class TestContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(TestContrastiveDivergence.class);

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

}
