package nn.rbm.learn;

import cern.colt.function.tdouble.DoubleFunction;
import data.mnist.MNISTImageLoader;
import math.DenseMatrix;
import math.Matrix;
import math.functions.Round;
import nn.rbm.RBM;
import nn.rbm.factory.RandomRBMFactory;
import org.apache.log4j.Logger;
import org.junit.Test;
import utils.PrettyPrint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by kenny on 5/27/14.
 */
public class TestRecurrentContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(TestRecurrentContrastiveDivergence.class);

    private static final RandomRBMFactory RBM_FACTORY = new RandomRBMFactory();

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
    public void recurrentNumbersTMinus1() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final int memory = 1;

        final RBM rbm = RBM_FACTORY.build(imageDim + imageDim * memory, 40); // two times the input because of the recurrent input
        final RecurrentContrastiveDivergence recurrentContrastiveDivergence = new RecurrentContrastiveDivergence(new LearningParameters().setEpochs(1000).setLearningRate(0.75).setMemory(memory));

        final List<Matrix> trainingData = new ArrayList<>();
        trainingData.add(DenseMatrix.make(totalDataSet.row(0)));      // 5
        trainingData.add(DenseMatrix.make(totalDataSet.row(1)));      // 0
        trainingData.add(DenseMatrix.make(totalDataSet.row(2)));      // 4
        trainingData.add(DenseMatrix.make(totalDataSet.row(3)));      // 1
        trainingData.add(DenseMatrix.make(totalDataSet.row(4)));      // 9
        trainingData.add(DenseMatrix.make(totalDataSet.row(5)));      // 2

        for(Matrix data : trainingData) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(data.row(0).toArray(), 28, 0.5));
        }

        recurrentContrastiveDivergence.learn(rbm, trainingData);

        // see if network consecutively draws numbers
        final DoubleFunction round = new Round(0.6);
        Matrix hidden;
        Matrix visual = trainingData.get(0); // supply first number to sequence, it'll use v_t to guess v_t+1
        LOGGER.info("Input : " + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
        for(int i = 0; i < trainingData.size() * 2; i++) { // make it loops twice
            hidden = recurrentContrastiveDivergence.runVisible(rbm, visual);
            visual = recurrentContrastiveDivergence.runHidden(rbm, hidden);

            LOGGER.info("Guess of what comes next\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.6));
            visual = DenseMatrix.make(visual.data().viewPart(0, imageDim, 1, imageDim * memory)); // trim off the previous input and only pass on the prediction
            visual.apply(round);

        }
    }


    @Test
    public void recurrentNumbersTMinusN() {
        final MNISTImageLoader mnistImageLoader = new MNISTImageLoader();
        final Matrix totalDataSet = mnistImageLoader.loadIdx3("/data/train-images-idx3-ubyte").divide(255.0);

        final int imageDim = totalDataSet.dim() / totalDataSet.rows(); // 784

        final int memory = 2;

        final RBM rbm = RBM_FACTORY.build(imageDim + imageDim * memory, 40); // two times the input because of the recurrent input
        final RecurrentContrastiveDivergence recurrentContrastiveDivergence = new RecurrentContrastiveDivergence(new LearningParameters().setEpochs(2500).setLearningRate(0.75).setMemory(memory));

        // two consecutive numbers, a t-1 recurrent network could not learn this, and loop
        final List<Matrix> trainingData = new ArrayList<>();
        trainingData.add(DenseMatrix.make(totalDataSet.row(0)));      // 5
        trainingData.add(DenseMatrix.make(totalDataSet.row(0)));      // 5
        trainingData.add(DenseMatrix.make(totalDataSet.row(2)));      // 4
        trainingData.add(DenseMatrix.make(totalDataSet.row(2)));      // 4
        trainingData.add(DenseMatrix.make(totalDataSet.row(4)));      // 9
        trainingData.add(DenseMatrix.make(totalDataSet.row(4)));      // 9
        trainingData.add(DenseMatrix.make(totalDataSet.row(0)));      // 5
        trainingData.add(DenseMatrix.make(totalDataSet.row(0)));      // 5

        for(Matrix data : trainingData) {
            LOGGER.info("\n" + PrettyPrint.toPixelBox(data.row(0).toArray(), 28, 0.5));
        }

        recurrentContrastiveDivergence.learn(rbm, trainingData);

        // see if network consecutively draws numbers
        final DoubleFunction round = new Round(0.6);
        Matrix hidden;
        Matrix visual = trainingData.get(0);
        for(int i = 1; i < memory; i++) {
            visual = visual.addColumns(trainingData.get(i));
        }
        LOGGER.info("Input : " + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.5));
        for(int i = 0; i < trainingData.size() * 2; i++) { // make it loops twice
            hidden = recurrentContrastiveDivergence.runVisible(rbm, visual);
            visual = recurrentContrastiveDivergence.runHidden(rbm, hidden);

            LOGGER.info("Guess of what comes next\n" + PrettyPrint.toPixelBox(visual.row(0).toArray(), 28, 0.6));
            visual = DenseMatrix.make(visual.data().viewPart(0, imageDim, 1, imageDim * memory)); // trim off the previous input and only pass on the prediction
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
            LOGGER.info("Guess of what comes next: " + PrettyPrint.toPixelBox(visual.toArray(), 0.5));
            visual.apply(round);
        }
    }

}
