package nn.rbm.learn;

import data.image.Image;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.encode.Matrix24BitImageEncoder;
import data.mnist.MNISTImageLoader;
import math.DenseMatrix;
import math.Matrix;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import org.apache.log4j.Logger;
import org.junit.Test;
import utils.PrettyPrint;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/27/14.
 */
public class TestDeepContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(TestDeepContrastiveDivergence.class);

    private static final RandomRBMFactory RBM_FACTORY = new RandomRBMFactory();

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


}



