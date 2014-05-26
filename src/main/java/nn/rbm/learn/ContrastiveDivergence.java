package nn.rbm.learn;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import math.DenseMatrix;
import math.Matrix;
import math.functions.Sigmoid;
import math.functions.doubledouble.rbm.ActivationState;
import nn.rbm.RBM;
import org.apache.log4j.Logger;
import utils.Clock;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by kenny on 5/15/14.
 *
 *  * http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
 */
public class ContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(ContrastiveDivergence.class);

    private static final DoubleDoubleFunction ACTIVATION_STATE = new ActivationState();

    private final Clock clock = new Clock();

    private final LearningParameters learningParameters;

    private final DoubleFunction logisticsFunction;

    public ContrastiveDivergence(final LearningParameters learningParameters) {
        this.learningParameters = learningParameters;
        this.logisticsFunction = new Sigmoid();
    }

    /**
     * Learn a matrix of data. Each row represents a single training set.
     * For example t = [[1,0],[0,1]] will train the rbm to recognize [1,0] and [0,1]
     * If a Matrix is too large I recommend splitting it into smaller matrices, but if they are reasonably small, this
     * is a fast way to "simultaneously" train multiple inputs and should generable be used unless the matrices are just
     * too large
     * @param rbm
     * @param dataSet
     */
    public void learn(final RBM rbm, final Matrix dataSet) {
        learn(rbm, Arrays.asList(dataSet));
    }

    public void learn(final RBM rbm, final Collection<Matrix> dataSets) {
        final Matrix weights = rbm.getWeights();

        clock.start();
        for(int epoch = 0; epoch < learningParameters.getEpochs(); epoch++) {

            double error = 0;
            for(Matrix dataSet : dataSets) {
                final int numberSamples = dataSet.rows();

                // Read training data and sample from the hidden later, positive CD phase, (reality phase)
                final Matrix positiveHiddenActivations = dataSet.dot(weights);
                final Matrix positiveHiddenProbabilities = positiveHiddenActivations.copy().apply(logisticsFunction);
                final Matrix positiveHiddenStates = positiveHiddenProbabilities.apply(DenseMatrix.random(numberSamples, rbm.getHiddenSize()), ACTIVATION_STATE);

                // Note that we're using the activation *probabilities* of the hidden states, not the hidden states themselves, when computing associations.
                // We could also use the states; see section 3 of Hinton's A Practical Guide to Training Restricted Boltzmann Machines" for more.
                final Matrix positiveAssociations = dataSet.transpose().dot(positiveHiddenProbabilities);

                // Reconstruct the visible units and sample again from the hidden units. negative CD phase, aka the daydreaming phase.
                final Matrix negativeVisibleActivations = positiveHiddenStates.dot(weights.transpose());
                final Matrix negativeVisibleProbabilities = negativeVisibleActivations.apply(logisticsFunction);
                final Matrix negativeHiddenActivations = negativeVisibleProbabilities.dot(weights);
                final Matrix negativeHiddenProbabilities = negativeHiddenActivations.apply(logisticsFunction);

                // Note, again, that we're using the activation *probabilities* when computing associations, not the states themselves.
                final Matrix negativeAssociations = negativeVisibleProbabilities.transpose().dot(negativeHiddenProbabilities);

                // Update weights.
                final Matrix updates = positiveAssociations.subtract(negativeAssociations).divide(numberSamples).multiply(learningParameters.getLearningRate());
                weights.add(updates);

                error += dataSet.copy().subtract(negativeVisibleProbabilities).pow(2).sum();
            }

            if(learningParameters.isLog() && epoch % 10 == 0 & epoch > 0) {
                LOGGER.info("Epoch: " + epoch + "/" + learningParameters.getEpochs() + ", error: " + error + ", time: " + clock.elapsedMillis() + "ms");
            }
            clock.reset();
        }
    }

    /*
        Assuming the FastRBM has been trained, run the network on a set of visible units to get a sample of the hidden units.
        Parameters, A matrix where each row consists of the states of the visible units.
        hidden_states, A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
     */
    public Matrix runVisible(final RBM rbm, final Matrix dataSet) {
        final int numberSamples = dataSet.rows();
        final Matrix weights = rbm.getWeights();

        // Calculate the activations of the hidden units.
        final Matrix hiddenActivations = dataSet.dot(weights);
        // Calculate the probabilities of turning the hidden units on.
        final Matrix hiddenProbabilities = hiddenActivations.apply(logisticsFunction);
        // Turn the hidden units on with their specified probabilities.
        final Matrix hiddenStates = hiddenProbabilities.apply(DenseMatrix.random(numberSamples, rbm.getHiddenSize()), ACTIVATION_STATE);

        return hiddenStates;
    }

    /*
        Assuming the FastRBM has been trained, run the network on a set of hidden units to get a sample of the visible units.
        Parameters, A matrix where each row consists of the states of the hidden units.
        visible_states, A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
     */
    public Matrix runHidden(final RBM rbm, final Matrix dataSet) {
        final int numberSamples = dataSet.rows();
        final Matrix weights = rbm.getWeights();

        // Calculate the activations of the hidden units.
        final Matrix visibleActivations = dataSet.dot(weights.transpose());
        // Calculate the probabilities of turning the visible units on.
        final Matrix visibleProbabilities = visibleActivations.apply(this.logisticsFunction);
        // Turn the visible units on with their specified probabilities.
        final Matrix visibleStates = visibleProbabilities.apply(DenseMatrix.random(numberSamples, rbm.getVisibleSize()), ACTIVATION_STATE);

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
        final Matrix weights = rbm.getWeights();

        // Take the first sample from a uniform distribution.
//        Matrix sample = DenseMatrix.make(new double[][]{dataSet.row(RANDOM.nextInt(numberSamples)).toArray()});
        Matrix sample = dataSet;

        // store all samples history
        final Set<Matrix> samples = new HashSet<>();

        // Start the alternating Gibbs sampling.
        // Note that we keep the hidden units binary states, but leave the visible units as real probabilities.
        // See section 3 of Hinton's "A Practical Guide to Training Restricted Boltzmann Machines" for more on why.
        for(int i = 0; i < dreamSamples; i++) {

            // Calculate the activations of the hidden units.
            final Matrix visibleValues = sample;
            samples.add(visibleValues);

            final Matrix hiddenActivations = visibleValues.dot(weights);
            // Calculate the probabilities of turning the hidden units on.
            final Matrix hiddenProbabilities = hiddenActivations.apply(this.logisticsFunction);
            // Turn the hidden units on with their specified probabilities.
            final Matrix hiddenStates = hiddenProbabilities.apply(DenseMatrix.random(sample.rows(), rbm.getHiddenSize()), ACTIVATION_STATE);

            // Calculate the activations of the hidden units.
            final Matrix visibleActivations = hiddenStates.dot(weights.transpose());
            // Calculate the probabilities of turning the visible units on.
            final Matrix visibleProbabilities = visibleActivations.apply(this.logisticsFunction);
            // Turn the visible units on with their specified probabilities.
            final Matrix visibleStates = visibleProbabilities.apply(DenseMatrix.random(sample.rows(), sample.columns()), ACTIVATION_STATE);

            sample = visibleStates;
        }
        return samples;
    }

}
