package nn.rbm.learn;

import com.google.common.base.Function;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
import nn.rbm.RBM;
import org.apache.log4j.Logger;
import utils.Clock;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Created by kenny on 5/15/14.
 *
 */
public class RecurrentContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(RecurrentContrastiveDivergence.class);

    private static final Random RANDOM = new Random();

    private static final Clock CLOCK = Clock.getInstance();

    private final LearningParameters learningParameters;

    private final Function<Double, Double> logisticsFunction;

    public RecurrentContrastiveDivergence(final LearningParameters learningParameters) {
        this.learningParameters = learningParameters;
        this.logisticsFunction = learningParameters.getLogisticsFunction();
    }

    public void learn(final RBM rbm, final Matrix dataSet) {
        final int numberSamples = dataSet.rows();
        rbm.addVisibleNodes(rbm.getHiddenSize());
        final Matrix weights = rbm.getWeights();
        LOGGER.info("weights 0: " + weights.rows() + "x" + weights.cols());

        LOGGER.info("Start Learning (" + numberSamples + " samples)");
        CLOCK.start();
        for(int epoch = 0; epoch < learningParameters.getEpochs(); epoch++) {

            final Matrix recurrentDataSet = dataSet.appendColumns(weights);
            // Read training data and sample from the hidden later, positive CD phase, (reality phase)
            final Matrix positiveHiddenActivations = recurrentDataSet.dot(weights);

            final Matrix positiveHiddenProbabilities = positiveHiddenActivations.apply(logisticsFunction);
            final Matrix random = ImmutableMatrix.random(numberSamples, rbm.getHiddenSize());
            final Matrix positiveHiddenStates = buildStatesFromActivationsMatrix(positiveHiddenProbabilities, random);

            // Note that we're using the activation *probabilities* of the hidden states, not the hidden states themselves, when computing associations.
            // We could also use the states; see section 3 of Hinton's A Practical Guide to Training Restricted Boltzmann Machines" for more.
            final Matrix positiveAssociations = recurrentDataSet.transpose().dot(positiveHiddenProbabilities);

            // Reconstruct the visible units and sample again from the hidden units. negative CD phase, aka the daydreaming phase.
            final Matrix negativeVisibleActivations = positiveHiddenStates.dot(ImmutableMatrix.transpose(weights));
            final Matrix negativeVisibleProbabilities = negativeVisibleActivations.apply(logisticsFunction);

            final Matrix negativeHiddenActivations = negativeVisibleProbabilities.dot(weights);
            final Matrix negativeHiddenProbabilities = negativeHiddenActivations.apply(logisticsFunction);

            // Note, again, that we're using the activation *probabilities* when computing associations, not the states themselves.
            final Matrix negativeAssociations = negativeVisibleProbabilities.transpose().dot(negativeHiddenProbabilities);

            // Update weights.
            weights.add(positiveAssociations.subtract(negativeAssociations).divide(numberSamples).multiply(learningParameters.getLearningRate()));

            final double error = recurrentDataSet.subtract(negativeVisibleProbabilities).pow(2).sum();

            if(epoch % 10 == 0) {
                LOGGER.info("Epoch: " + epoch + "/" + learningParameters.getEpochs() + ", error: " + error + ", time: " + CLOCK.elapsedMillis() + "ms");
                CLOCK.reset();
            }
        }
    }

    /*
        Assuming the RBM has been trained, run the network on a set of visible units to get a sample of the hidden units.
        Parameters, A matrix where each row consists of the states of the visible units.
        hidden_states, A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
     */
    public Matrix runVisible(final RBM rbm, final Matrix dataSet) {
        final int numberSamples = dataSet.rows();
        rbm.addVisibleNodes(rbm.getHiddenSize());
        final Matrix weights = rbm.getWeights();
        final Matrix recurrentDataSet = dataSet.appendColumns(weights);

        // Calculate the activations of the hidden units.
        final Matrix hiddenActivations = recurrentDataSet.dot(weights);
        // Calculate the probabilities of turning the hidden units on.
        final Matrix hiddenProbabilities = hiddenActivations.apply(logisticsFunction);
        // Turn the hidden units on with their specified probabilities.
        final Matrix hiddenStates = buildStatesFromActivationsMatrix(hiddenProbabilities, ImmutableMatrix.random(numberSamples, rbm.getHiddenSize()));

        return hiddenStates;
    }

    /*
        Assuming the RBM has been trained, run the network on a set of hidden units to get a sample of the visible units.
        Parameters, A matrix where each row consists of the states of the hidden units.
        visible_states, A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
     */
    public Matrix runHidden(final RBM rbm, final Matrix dataSet) {
        final int numberSamples = dataSet.rows();
        rbm.addVisibleNodes(rbm.getHiddenSize());
        final Matrix weights = rbm.getWeights();

        // Calculate the activations of the hidden units.
        final Matrix visibleActivations = dataSet.dot(ImmutableMatrix.transpose(weights));
        // Calculate the probabilities of turning the visible units on.
        final Matrix visibleProbabilities = visibleActivations.apply(this.logisticsFunction);
        // Turn the visible units on with their specified probabilities.
        final Matrix visibleStates = buildStatesFromActivationsMatrix(visibleProbabilities, ImmutableMatrix.random(numberSamples, rbm.getVisibleSize() + rbm.getHiddenSize()));

        return visibleStates;
    }

    /*
        Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.
        Note that we only initialize the network *once*, so these samples are correlated.
        samples: A matrix, where each row is a sample of the visible units produced while the network was daydreaming.
     */
    public Set<Matrix> dayDream(final RBM rbm, final Matrix dataSet, final int dreamSamples) {
        final int numberSamples = dataSet.rows();
        rbm.addVisibleNodes(rbm.getHiddenSize());
        final Matrix weights = rbm.getWeights();
        final Matrix recurrentDataSet = dataSet.appendColumns(weights);

        // Take the first sample from a uniform distribution.
        double[] sample = recurrentDataSet.data()[RANDOM.nextInt(numberSamples)];

        // store all samples history
        final Set<Matrix> samples = new HashSet<>();

        // Start the alternating Gibbs sampling.
        // Note that we keep the hidden units binary states, but leave the visible units as real probabilities.
        // See section 3 of Hinton's "A Practical Guide to Training Restricted Boltzmann Machines" for more on why.
        for(int i = 0; i < dreamSamples; i++) {

            // Calculate the activations of the hidden units.
            final Matrix visibleValues = new ImmutableMatrix(sample);
            samples.add(visibleValues);

            final Matrix hiddenActivations = visibleValues.dot(weights);
            // Calculate the probabilities of turning the hidden units on.
            final Matrix hiddenProbabilities = hiddenActivations.apply(this.logisticsFunction);
            // Turn the hidden units on with their specified probabilities.
            final Matrix hiddenStates = buildStatesFromActivationsMatrix(hiddenProbabilities, ImmutableMatrix.random(numberSamples, rbm.getHiddenSize()));

            // Calculate the activations of the hidden units.
            final Matrix visibleActivations = hiddenStates.dot(ImmutableMatrix.transpose(weights));
            // Calculate the probabilities of turning the visible units on.
            final Matrix visibleProbabilities = visibleActivations.apply(this.logisticsFunction);
            // Turn the visible units on with their specified probabilities.
            final Matrix visibleStates = buildStatesFromActivationsMatrix(visibleProbabilities, ImmutableMatrix.random(numberSamples, sample.length));

            sample = visibleStates.data()[0];
        }
        return samples;
    }

    private static Matrix buildStatesFromActivationsMatrix(final Matrix activationMatrix, Matrix randomStateMatrix) {
        final int rows = activationMatrix.rows();
        final int cols = activationMatrix.cols();
        final double[][] stateMatrix = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                stateMatrix[i][j] = activationMatrix.get(i, j) >= randomStateMatrix.get(i, j) ? 1.0 : 0.0;
            }
        }
        return new ImmutableMatrix(stateMatrix);
    }

}
