package nn.rbm.learn;


import cern.colt.function.tdouble.DoubleFunction;
import math.DenseMatrix;
import math.Matrix;
import math.functions.OneMinusX;
import math.functions.Sigmoid;
import nn.rbm.RBM;
import org.apache.log4j.Logger;

import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class BackErrorPropagation {

    private static final Logger LOGGER = Logger.getLogger(BackErrorPropagation.class);

    private static final DoubleFunction ONE_MINUS_X = new OneMinusX();

    private static final DoubleFunction SIGMOID = new Sigmoid();

    private final LearningParameters learningParameters;

    public BackErrorPropagation(final LearningParameters learningParameters) {
        this.learningParameters = learningParameters;
    }

    public double learn(final RBM rbm, final List<Matrix> trainData, final List<Matrix> teacherSignals) {
        double error = 0;
        for(int epoch = 0; epoch < learningParameters.getEpochs(); epoch++) {
            error = 0;
            for(int i = 0; i < trainData.size(); i++) {
                final Matrix input = trainData.get(i).copy();
                final Matrix teacherSignal = teacherSignals.get(i).copy();

                Matrix output = feedFoward(rbm, input);
                error += calculateAvgSquaredError(output, teacherSignal);
                Matrix errors = calculateErrors(output, teacherSignal);
                adjustWeights(rbm, input, errors);

            }
            if(learningParameters.isLog() && epoch > 0 && epoch % 10 == 0) {
                LOGGER.info("Epoch: " + epoch + ", error: " + error / trainData.size());
            }
        }
        return error;
    }
    public double backPropagate(final RBM rbm, final Matrix output, final Matrix teacherSignals) {
        Matrix errors = calculateErrors(output, teacherSignals);
        adjustWeights(rbm, output, errors);
        return calculateAvgSquaredError(output, teacherSignals);
    }

    /**
     * feed forward
     */
    public Matrix feedFoward(final RBM rbm, final Matrix input) {
        final Matrix output = DenseMatrix.make(1, rbm.getHiddenSize());

        final Matrix weights = rbm.getWeights();

        for (int i = 0; i < rbm.getVisibleSize(); i++) {
            for (int j = 0; j < rbm.getHiddenSize(); j++) {
                output.set(0, j, output.get(0, j) + input.get(0, i) * weights.get(i, j));
            }
        }
        return output.apply(SIGMOID);
    }


    /**
     * calculate the error
     */
    private Matrix calculateErrors(final Matrix output, final Matrix teacherSignals) {
        // (teacher_i - output_i)  * output_i * (1 - output_i)
        final Matrix errors = teacherSignals.copy().subtract(output).multiply(output).multiply(output.apply(ONE_MINUS_X));
        return errors;
    }

    /**
     * depending on the error, adjust the weights
     */
    private void adjustWeights(final RBM rbm, final Matrix input, final Matrix errors) {
        final Matrix weights = rbm.getWeights();
        //  adjust the weights
        for (int i = 0; i < rbm.getVisibleSize(); i++) {
            for (int j = 0; j < rbm.getHiddenSize(); j++) {
                weights.set(i, j, weights.get(i, j) + learningParameters.getLearningRate() * errors.get(0, j) * input.get(0, i));
            }
        }
    }

    /**
     * calculate the average squared error between the
     * output layer and teacher signal
     *
     */
    private double calculateAvgSquaredError(final Matrix output, final Matrix teacherSignals) {
        return output.copy().subtract(teacherSignals).pow(2).sum() / output.columns();
    }
}
