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

import java.util.List;

/**
 * Created by kenny on 5/15/14.
 *
 */
public class RecurrentContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(RecurrentContrastiveDivergence.class);

    private static final DoubleDoubleFunction ACTIVATION_STATE = new ActivationState();

    private static final DoubleFunction SIGMOID = new Sigmoid();

    private static final Clock CLOCK = new Clock();

    private final LearningParameters learningParameters;

    public RecurrentContrastiveDivergence(final LearningParameters learningParameters) {
        this.learningParameters = learningParameters;
    }

    /**
     * learn a sequence of events
     * @param rbm
     * @param events
     */
    public void learn(final RBM rbm, final List<Matrix> events) {
        final int numberEvents = events.size();
        final Matrix weights = rbm.getWeights();

        LOGGER.info("Start Learning (" + numberEvents + " events)");
        CLOCK.start();
        for(int epoch = 0; epoch < learningParameters.getEpochs(); epoch++) {

            Matrix previousEvent = DenseMatrix.make(events.get(0).rows(), events.get(0).columns());

            for(Matrix event : events) {
                final Matrix recurrentDataSet = event.addColumns(previousEvent);
                // LOGGER.info("Recurrent Dataset\n" + PrettyPrint.toPixelBox(recurrentDataSet.toArray(), 0.5));

                // Read training data and sample from the hidden later, positive CD phase, (reality phase)
                final Matrix positiveHiddenActivations = recurrentDataSet.dot(weights);
                final Matrix positiveHiddenProbabilities = positiveHiddenActivations.apply(SIGMOID);
                final Matrix positiveHiddenStates = positiveHiddenProbabilities.copy().apply(DenseMatrix.random(event.rows(), rbm.getHiddenSize()), ACTIVATION_STATE);

                // Note that we're using the activation *probabilities* of the hidden states, not the hidden states themselves, when computing associations.
                // We could also use the states; see section 3 of Hinton's A Practical Guide to Training Restricted Boltzmann Machines" for more.
                final Matrix positiveAssociations = recurrentDataSet.transpose().dot(positiveHiddenProbabilities);

                // Reconstruct the visible units and sample again from the hidden units. negative CD phase, aka the daydreaming phase.
                final Matrix negativeVisibleActivations = positiveHiddenStates.dot(weights.transpose());
                final Matrix negativeVisibleProbabilities = negativeVisibleActivations.apply(SIGMOID);
                final Matrix negativeHiddenActivations = negativeVisibleProbabilities.dot(weights);
                final Matrix negativeHiddenProbabilities = negativeHiddenActivations.apply(SIGMOID);

                // Note, again, that we're using the activation *probabilities* when computing associations, not the states themselves.
                final Matrix negativeAssociations = negativeVisibleProbabilities.transpose().dot(negativeHiddenProbabilities);

                // Update weights.
                weights.add(positiveAssociations.subtract(negativeAssociations).divide(numberEvents).multiply(learningParameters.getLearningRate()));

                final double error = recurrentDataSet.subtract(negativeVisibleProbabilities).pow(2).sum();

                previousEvent = event;
                if(epoch % 10 == 0) {
                    LOGGER.info("Epoch: " + epoch + "/" + learningParameters.getEpochs() + ", error: " + error + ", time: " + CLOCK.elapsedMillis() + "ms");
                    CLOCK.reset();
                }
            }
        }
    }

    /*
        Assuming the RBM has been trained, run the network on a set of visible units to get a sample of the hidden units.
        Parameters, A matrix where each row consists of the states of the visible units.
        hidden_states, A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.

        Recurrent version pass in an empty t-1 visible layer
     */
    public Matrix runVisible(final RBM rbm, final Matrix dataSet) {
        final int numberSamples = dataSet.rows();

        final Matrix weights = rbm.getWeights();

        final Matrix recurrentDataSet = dataSet.addColumns(DenseMatrix.make(dataSet.rows(), dataSet.columns())); // append an empty t-1 visible layer

        // Calculate the activations of the hidden units.
        final Matrix hiddenActivations = recurrentDataSet.dot(weights);
        // Calculate the probabilities of turning the hidden units on.
        final Matrix hiddenProbabilities = hiddenActivations.apply(SIGMOID);
        // Turn the hidden units on with their specified probabilities.
        final Matrix hiddenStates = hiddenProbabilities.apply(DenseMatrix.random(numberSamples, rbm.getHiddenSize()), ACTIVATION_STATE);

        return hiddenStates;
    }

    /*
        Assuming the RBM has been trained, run the network on a set of hidden units to get a sample of the visible units.
        Parameters, A matrix where each row consists of the states of the hidden units.
        visible_states, A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
     */
    public Matrix runHidden(final RBM rbm, final Matrix dataSet) {

        final Matrix weights = rbm.getWeights();

        // Calculate the activations of the hidden units.
        final Matrix visibleActivations = dataSet.dot(weights.transpose());
        // Calculate the probabilities of turning the visible units on.
        final Matrix visibleProbabilities = visibleActivations.apply(SIGMOID);
        // Turn the visible units on with their specified probabilities.
        final Matrix visibleStates = visibleProbabilities.apply(DenseMatrix.random(dataSet.rows(), rbm.getVisibleSize()), ACTIVATION_STATE);

        return visibleStates;
    }



}
