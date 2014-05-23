package nn.rbm.learn;

import com.google.common.base.Function;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import math.matrix.MutableMatrix;
import nn.rbm.RBM;
import org.apache.log4j.Logger;

import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class BackErrorPropagation {

    private static final Logger LOGGER = Logger.getLogger(BackErrorPropagation.class);

    private final LearningParameters learningParameters;

    private final ContrastiveDivergence contrastiveDivergence;

    public BackErrorPropagation(final LearningParameters learningParameters) {
        this.learningParameters = learningParameters;
        this.contrastiveDivergence = new ContrastiveDivergence(learningParameters);
    }

    private static final Function<Double, Double> ONE_MINUS_X = new Function<Double, Double>() {
        @Override
        public Double apply(Double x) {
            return 1.0 - x;
        }
    };

    public double learn(final RBM rbm, final List<Matrix> trainData, final List<Matrix> teacherSignals) {
        double error = 0;
        for(int epoch = 0; epoch < learningParameters.getEpochs(); epoch++) {
            error = 0;
            for(int i = 0; i < trainData.size(); i++) {
                final Matrix data = trainData.get(i);
                final Matrix teacherSignal = teacherSignals.get(i);

                Matrix output = feedFoward(rbm, data);
                error += backPropagate(rbm, output, teacherSignal);
            }
            if(learningParameters.isLog() && epoch > 0 && epoch % 10 == 0) {
                LOGGER.info("Epoch: " + epoch + ", error: " + error);
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
        final Matrix output = new MutableMatrix(1, rbm.getHiddenSize());

        final Matrix weights = rbm.getWeights();
        for (int i = 0; i < rbm.getVisibleSize(); i++) {
            for (int j = 0; j < rbm.getHiddenSize(); j++) {
                output.set(0, j, output.get(0, j) + input.get(0, i) * weights.get(i, j));
            }
        }
        return new ImmutableMatrix(output).apply(learningParameters.getLogisticsFunction());
    }


    /**
     * calculate the error
     */
    private Matrix calculateErrors(final Matrix output, final Matrix teacherSignals) {
        // wrap in immutable as outputs is mutable in other parts of the code
        final Matrix oneMinusOutput = new ImmutableMatrix(output).apply(ONE_MINUS_X);
        // (teacher_i - output_i)  * output_i * (1 - output_i)
        return teacherSignals.subtract(output).multiply(output).multiply(oneMinusOutput);
    }

    /**
     * depending on the error, adjust the weights
     */
    private void adjustWeights(final RBM rbm, final Matrix output, final Matrix errors) {
        final Matrix weights = rbm.getWeights();
        //  adjust the weights
        for (int i = 0; i < rbm.getVisibleSize(); i++) {
            for (int j = 0; j < rbm.getHiddenSize(); j++) {
                weights.set(i, j, weights.get(i, j) + learningParameters.getLearningRate() * errors.get(0, j) * output.get(0, j));
            }
        }
    }

    /**
     * calculate the average squared error between the
     * output layer and teacher signal
     *
     */
    private double calculateAvgSquaredError(final Matrix output, final Matrix teacherSignals) {
        return output.subtract(teacherSignals).pow(2).sum() / output.cols();
    }
}
