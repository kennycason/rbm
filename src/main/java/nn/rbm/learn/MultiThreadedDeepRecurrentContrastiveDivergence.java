package nn.rbm.learn;


import math.DenseMatrix;
import math.Matrix;
import nn.rbm.RBM;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.RBMLayer;
import org.apache.log4j.Logger;
import utils.Clock;
import utils.concurrent.ThreadPoolExecutor;

import java.util.List;

/**
 * Created by kenny on 5/22/14.
 *
 */
public class MultiThreadedDeepRecurrentContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(MultiThreadedDeepRecurrentContrastiveDivergence.class);

    private final Clock clock = new Clock();

    final ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor();

    private final DeepContrastiveDivergence deepContrastiveDivergence;

    private final ContrastiveDivergence contrastiveDivergence;

    private final LearningParameters learningParameters;

    public MultiThreadedDeepRecurrentContrastiveDivergence(final LearningParameters learningParameters) {
        this(learningParameters, 8);
    }

    public MultiThreadedDeepRecurrentContrastiveDivergence(final LearningParameters learningParameters, final int numberThreads) {
        this.contrastiveDivergence = new ContrastiveDivergence(learningParameters);
        this.deepContrastiveDivergence = new DeepContrastiveDivergence(learningParameters);
        this.learningParameters = learningParameters;
        this.threadPoolExecutor.setNumThreads(numberThreads);
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
                threadPoolExecutor.add(new ContrastiveDivergenceRunner(rbm, splitDataSet, this.learningParameters));
            }
            threadPoolExecutor.execute();
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
        return this.deepContrastiveDivergence.runVisible(deepRBM, dataSet);
    }

    /*
        Assuming the RBM has been trained, run the network on a set of hidden units to get a sample of the visible units.
        Parameters, A matrix where each row consists of the states of the hidden units.
        visible_states, A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
     */
    public Matrix runHidden(final DeepRBM deepRBM, final Matrix dataSet) {
        return this.deepContrastiveDivergence.runHidden(deepRBM, dataSet);
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

    private static class ContrastiveDivergenceRunner implements Runnable {

        private final ContrastiveDivergence contrastiveDivergence;

        private final Matrix dataSet;

        private final RBM rbm;

        public ContrastiveDivergenceRunner(final RBM rbm, final Matrix dataSet, final LearningParameters learningParameters) {
            this.contrastiveDivergence = new ContrastiveDivergence(learningParameters);
            this.rbm = rbm;
            this.dataSet = dataSet;
        }

        @Override
        public void run() {
            try {
                this.contrastiveDivergence.learn(rbm, dataSet);
            } catch(Exception e) {
                e.printStackTrace();
            }
        }

    }

}
