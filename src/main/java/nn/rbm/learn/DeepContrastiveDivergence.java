package nn.rbm.learn;

import com.google.common.base.Function;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;
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

    private final ContrastiveDivergence contrastiveDivergence;

    private static final Clock CLOCK = Clock.getInstance();

    private final LearningParameters learningParameters;

    private final Function<Double, Double> logisticsFunction;

    public DeepContrastiveDivergence(final LearningParameters learningParameters) {
        this.learningParameters = learningParameters;
        this.logisticsFunction = learningParameters.getLogisticsFunction();
        this.contrastiveDivergence = new ContrastiveDivergence(this.learningParameters);
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
        CLOCK.start();

        for(int layer = 0; layer < rbmLayers.length; layer++) {
            LOGGER.info("Training Layer: " + layer);

            final RBMLayer rbmLayer = rbmLayers[layer];
            samplePieces = buildSamplesFromActivatedHiddenLayers(samplePieces, layer, rbmLayers);

            for(int r = 0; r < rbmLayer.size(); r++) {
                final RBM rbm = rbmLayer.getRBM(r);
                final Matrix splitDataSet = samplePieces.get(r);
                this.contrastiveDivergence.learn(rbm, splitDataSet);
            }
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
        double[][][] hiddenStatesArray = new double[0][0][0];

        for(int layer = 0; layer < rbmLayers.length; layer++) {
            final RBMLayer rbmLayer = rbmLayers[layer];
            hiddenStatesArray = new double[rbmLayer.size()][][];

            samplePieces = buildSampleData(samplePieces, layer, rbmLayers);

            for(int r = 0; r < rbmLayer.size(); r++) {
                final RBM rbm = rbmLayer.getRBM(r);
                final Matrix splitDataSet = samplePieces.get(r);
                final Matrix hiddenStates = this.contrastiveDivergence.runVisible(rbm, splitDataSet);
                hiddenStatesArray[r] = hiddenStates.data();
            }
        }

        return new ImmutableMatrix(Matrix.appendColumns(hiddenStatesArray));

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
        double[][][] visibleStatesArray = new double[0][0][0];

        for(int layer = rbmLayers.length - 1; layer >= 0; layer--) {
            final RBMLayer rbmLayer = rbmLayers[layer];
            visibleStatesArray = new double[rbmLayer.size()][][];

            samplePieces = buildSampleDataReverse(samplePieces, layer, rbmLayers);

            for(int r = 0; r < rbmLayer.size(); r++) {
                final RBM rbm = rbmLayer.getRBM(r);
                final Matrix splitDataSet = samplePieces.get(r);

                final Matrix visibleStates = this.contrastiveDivergence.runHidden(rbm, splitDataSet);
                visibleStatesArray[r] = visibleStates.data();
            }
        }
        return new ImmutableMatrix(Matrix.appendColumns(visibleStatesArray));
    }

    /*
        Pass data into visible layers and activate hidden layers.
        return hidden layers
     */
    private List<Matrix> buildSamplesFromActivatedHiddenLayers(List<Matrix> sampleData, int layer, RBMLayer[] rbmLayers) {
        final RBMLayer rbmLayer = rbmLayers[layer];

        if(layer == 0) {
            return sampleData;
        }
        else {
            final RBMLayer previousLayer = rbmLayers[layer - 1];
            double[][][] previousLayerOutputs = new double[previousLayer.size()][][];
            for(int r = 0; r < previousLayer.size(); r++) {
                final RBM rbm = previousLayer.getRBM(r);
                previousLayerOutputs[r] = this.contrastiveDivergence.runVisible(rbm, sampleData.get(r)).data();
            }
            // combine all outputs off hidden layer, then re-split them to input into the next visual layer
            return new ImmutableMatrix(Matrix.appendColumns(previousLayerOutputs)).splitColumns(rbmLayer.size()) ;
        }
    }

    private List<Matrix> buildSampleData(List<Matrix> trainingData, int layer, RBMLayer[] rbmLayers) {
        final RBMLayer rbmLayer = rbmLayers[layer];

        if(layer == 0) {
            return trainingData;
        }
        else {
            final RBMLayer previousLayer = rbmLayers[layer - 1];
            double[][][] previousLayerOutputs = new double[previousLayer.size()][][];
            for(int r = 0; r < previousLayer.size(); r++) {
                previousLayerOutputs[r] = this.contrastiveDivergence.runVisible(previousLayer.getRBM(r), trainingData.get(r)).data();
                       // previousLayer.getRBM(r).getHidden().getValues() };
            }
            // combine all outputs off hidden layer, then re-split them to input into the next visual layer
            return new ImmutableMatrix(Matrix.appendColumns(previousLayerOutputs)).splitColumns(rbmLayer.size()) ;
        }
    }

    private List<Matrix> buildSampleDataReverse(List<Matrix> trainingData, int layer, RBMLayer[] rbmLayers) {
        final RBMLayer rbmLayer = rbmLayers[layer];

        if(layer == rbmLayers.length - 1) {
            return trainingData;
        }
        else {
            final RBMLayer previousLayer = rbmLayers[layer + 1];
            double[][][] previousLayerInputs = new double[previousLayer.size()][][];
            for(int r = 0; r < previousLayer.size(); r++) {
                previousLayerInputs[r] = this.contrastiveDivergence.runHidden(previousLayer.getRBM(r), trainingData.get(r)).data();
            }
            // combine all outputs off hidden layer, then re-split them to input into the next visual layer
            return new ImmutableMatrix(Matrix.appendColumns(previousLayerInputs)).splitColumns(rbmLayer.size());
        }
    }

}
