package nn.rbm.learn;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import math.DenseMatrix;
import math.Matrix;
import math.functions.doubledouble.rbm.ActivationState;
import nn.rbm.RBM;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.RBMLayer;
import org.apache.log4j.Logger;
import utils.Clock;

import java.util.List;

/**
 * Created by kenny on 5/15/14.
 *
 */
public class DeepContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(DeepContrastiveDivergence.class);

    private static final DoubleDoubleFunction ACTIVATION_STATE_FUNCTION = new ActivationState();

    private final Clock clock = new Clock();

    private final ContrastiveDivergence contrastiveDivergence;

    private final LearningParameters learningParameters;

    public DeepContrastiveDivergence(final LearningParameters learningParameters) {
        this.contrastiveDivergence = new ContrastiveDivergence(learningParameters);
        this.learningParameters = learningParameters;
    }

    /*
       DBN Greedy Training
       P(v,h1,h2,...hn) = P(v|h1)P(h1|h2)...P(hn-2|hn-1)P(hn-1|hn)
       Train P(v|h1), use h1 for each v to train P(h1|h2), repeat until P(hn-1|hn) is trained
     */
    public void learn(final DeepRBM deepRBM, final Matrix dataSet) {
        final RBMLayer[] rbmLayers = deepRBM.getRbmLayers();

        final List<Matrix> trainingData = dataSet.splitColumns(rbmLayers[0].size()); // split dataset across rbms

        List<Matrix> samplePieces = trainingData;
        clock.reset();
        for(int layer = 0; layer < rbmLayers.length; layer++) {

            final RBMLayer rbmLayer = rbmLayers[layer];
            samplePieces = buildSamplesFromActivatedHiddenLayers(samplePieces, layer, rbmLayers);

            for(int r = 0; r < rbmLayer.size(); r++) {
                final RBM rbm = rbmLayer.getRBM(r);
                final Matrix splitDataSet = samplePieces.get(r);
                this.contrastiveDivergence.learn(rbm, splitDataSet);
            }

        }

        if(learningParameters.isLog()) {
            LOGGER.info("All Layers finished Training in " + clock.elapsedSeconds() + "ms");
        }
    }

    /*
        Assuming the RBM has been trained, run the network on a set of visible units to get a sample of the hidden units.
        Parameters, A matrix where each row consists of the states of the visible units.
        hidden_states, A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
     */
    public Matrix runVisible(final DeepRBM deepRBM, final Matrix dataSet) {
        final RBMLayer[] rbmLayers = deepRBM.getRbmLayers();

        final List<Matrix> trainingData = dataSet.splitColumns(rbmLayers[0].size()); // split dataset across rbms

        List<Matrix> samplePieces = trainingData;
        Matrix[] hiddenStatesArray = new Matrix[0];

        for(int layer = 0; layer < rbmLayers.length; layer++) {
            final RBMLayer rbmLayer = rbmLayers[layer];
            hiddenStatesArray = new Matrix[rbmLayer.size()];

            samplePieces = buildSampleData(samplePieces, layer, rbmLayers);

            for(int r = 0; r < rbmLayer.size(); r++) {
                final RBM rbm = rbmLayer.getRBM(r);
                final Matrix splitDataSet = samplePieces.get(r);
                final Matrix hiddenStates = this.contrastiveDivergence.runVisible(rbm, splitDataSet);
                hiddenStatesArray[r] = hiddenStates;
            }
        }

        return DenseMatrix.make(Matrix.concatColumns(hiddenStatesArray));

    }

    /*
        Assuming the RBM has been trained, run the network on a set of hidden units to get a sample of the visible units.
        Parameters, A matrix where each row consists of the states of the hidden units.
        visible_states, A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
     */
    public Matrix runHidden(final DeepRBM deepRBM, final Matrix dataSet) {
        final RBMLayer[] rbmLayers = deepRBM.getRbmLayers();

        final List<Matrix> trainingData = dataSet.splitColumns(rbmLayers[rbmLayers.length - 1].size()); // split dataset across rbms

        List<Matrix> samplePieces = trainingData;
        Matrix[] visibleStatesArray = new Matrix[0];

        for(int layer = rbmLayers.length - 1; layer >= 0; layer--) {
            final RBMLayer rbmLayer = rbmLayers[layer];
            visibleStatesArray = new Matrix[rbmLayer.size()];

            samplePieces = buildSampleDataReverse(samplePieces, layer, rbmLayers);

            for(int r = 0; r < rbmLayer.size(); r++) {
                final RBM rbm = rbmLayer.getRBM(r);
                final Matrix splitDataSet = samplePieces.get(r);

                final Matrix visibleStates = this.contrastiveDivergence.runHidden(rbm, splitDataSet);
                visibleStatesArray[r] = visibleStates;
            }
        }
        return DenseMatrix.make(Matrix.concatColumns(visibleStatesArray));
    }

    /*
        Pass data into visible layers and activate hidden layers.
        return hidden layers
     */
    private List<Matrix> buildSamplesFromActivatedHiddenLayers(final List<Matrix> sampleData, final int layer, RBMLayer[] rbmLayers) {
        final RBMLayer rbmLayer = rbmLayers[layer];

        if(layer == 0) {
            return sampleData;
        }
        else {
            final RBMLayer previousLayer = rbmLayers[layer - 1];
            Matrix[] previousLayerOutputs = new Matrix[previousLayer.size()];
            for(int r = 0; r < previousLayer.size(); r++) {
                final RBM rbm = previousLayer.getRBM(r);
                previousLayerOutputs[r] = this.contrastiveDivergence.runVisible(rbm, sampleData.get(r));
            }
            // combine all outputs off hidden layer, then re-split them to input into the next visual layer
            return DenseMatrix.make(Matrix.concatColumns(previousLayerOutputs)).splitColumns(rbmLayer.size());
        }
    }

    private List<Matrix> buildSampleData(final List<Matrix> trainingData, final int layer, final RBMLayer[] rbmLayers) {
        final RBMLayer rbmLayer = rbmLayers[layer];

        if(layer == 0) {
            return trainingData;
        }
        else {
            final RBMLayer previousLayer = rbmLayers[layer - 1];
            Matrix[] previousLayerOutputs = new Matrix[previousLayer.size()];
            for(int r = 0; r < previousLayer.size(); r++) {
                previousLayerOutputs[r] = this.contrastiveDivergence.runVisible(previousLayer.getRBM(r), trainingData.get(r));
                       // previousLayer.getRBM(r).getHidden().getValues() };
            }
            // combine all outputs off hidden layer, then re-split them to input into the next visual layer
            return DenseMatrix.make(Matrix.concatColumns(previousLayerOutputs)).splitColumns(rbmLayer.size());
        }
    }

    private List<Matrix> buildSampleDataReverse(final List<Matrix> trainingData, final int layer, final RBMLayer[] rbmLayers) {
        final RBMLayer rbmLayer = rbmLayers[layer];

        if(layer == rbmLayers.length - 1) {
            return trainingData;
        }
        else {
            final RBMLayer previousLayer = rbmLayers[layer + 1];
            Matrix[] previousLayerInputs = new Matrix[previousLayer.size()];
            for(int r = 0; r < previousLayer.size(); r++) {
                previousLayerInputs[r] = this.contrastiveDivergence.runHidden(previousLayer.getRBM(r), trainingData.get(r));
            }
            // combine all outputs off hidden layer, then re-split them to input into the next visual layer
            return DenseMatrix.make(Matrix.concatColumns(previousLayerInputs)).splitColumns(rbmLayer.size());
        }
    }

}
