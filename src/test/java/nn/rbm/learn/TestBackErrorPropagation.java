package nn.rbm.learn;

import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import nn.rbm.RBM;
import nn.rbm.factory.RandomRBMFactory;
import org.apache.log4j.Logger;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/23/14.
 */
public class TestBackErrorPropagation {

    private static final Logger LOGGER = Logger.getLogger(TestBackErrorPropagation.class);

    @Test
    public void backErrorPropagation() {
        final List<Matrix> trainData = new ArrayList<>();
//        trainData.add(new MutableMatrix(new double[] {0, 1}));
//        trainData.add(new ImmutableMatrix(new double[] {0, 1}));
//        trainData.add(new ImmutableMatrix(new double[] {1, 0}));
//        trainData.add(new ImmutableMatrix(new double[] {1, 1}));
        trainData.add(new ImmutableMatrix(new double[] {1, 0}));
 //       trainData.add(new ImmutableMatrix(new double[] {0, 1}));

        final List<Matrix> teacherSignals = new ArrayList<>();
//        teacherSignals.add(new MutableMatrix(new double[] {0}));
//        teacherSignals.add(new ImmutableMatrix(new double[] {0}));
//        teacherSignals.add(new ImmutableMatrix(new double[] {0}));
//        teacherSignals.add(new ImmutableMatrix(new double[] {1}));
        teacherSignals.add(new ImmutableMatrix(new double[] {0}));
  //      teacherSignals.add(new ImmutableMatrix(new double[] {1}));

        final RBM rbm = new RandomRBMFactory().build(2, 1);

        final LearningParameters learningParameters = new LearningParameters().setEpochs(500).setLearningRate(0.5);
        final BackErrorPropagation backErrorPropagation = new BackErrorPropagation(learningParameters);

        backErrorPropagation.learn(rbm, trainData, teacherSignals);
        for(Matrix data : trainData) {
            final Matrix output = backErrorPropagation.feedFoward(rbm, data);
            LOGGER.info(output);
        }

    }

}
