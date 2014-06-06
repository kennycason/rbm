package nn.rbm.deep;

import nn.rbm.RBM;
import nn.rbm.factory.RBMFactory;
import utils.PrettyPrint;

/**
 * Created by kenny on 5/16/14.
 *
 * A deep RBM can be configured in the standard way (one RBM per layer)
 *  h_2     (o   o   o)
 *            [X   X]
 *  h_1     (o   o   o)
 *            [X   X]
 *  v       (o   o   o)
 *
 * Or it can be configured as a Shape RBM (multiple RBMs per layer)
 *  h_2     (o   o   o   o)
 *            [X   X   X]
 *  h_1     (o   o) (o   o)
 *            [X   X   X]
 *  v       (o) (o) (o) (o)
 */
public class DeepRBM {

    private final RBMLayer[] rbmLayers;

    private final int visibleSize;

    private final int hiddenSize;

    public DeepRBM(final LayerParameters[] layerParameters, final RBMFactory rbmFactory) {
        checkLayerParameters(layerParameters);

        rbmLayers = new RBMLayer[layerParameters.length];

        this.visibleSize = layerParameters[0].getVisibleUnitsPerRBM() * layerParameters[0].getNumRBMS();
        this.hiddenSize = layerParameters[layerParameters.length - 1].getHiddenUnitsPerRBM() * layerParameters[layerParameters.length - 1].getNumRBMS();

        for(int layer = 0; layer < layerParameters.length; layer++) {
            final LayerParameters layerParameter = layerParameters[layer];
            final RBM[] rbms = new RBM[layerParameter.getNumRBMS()];

            for(int i = 0; i < layerParameter.getNumRBMS(); i++) {
                rbms[i] = rbmFactory.build(layerParameter.getVisibleUnitsPerRBM(), layerParameter.getHiddenUnitsPerRBM());
            }
            rbmLayers[layer] = new RBMLayer(rbms);
        }
    }

    public static void checkLayerParameters(final LayerParameters[] layerParameters) {

        //int visibleNodesIn = layerParameters[0].getNumRBMS() * layerParameters[0].getVisibleUnitsPerRBM();
        int hiddenNodesOut = layerParameters[0].getNumRBMS() * layerParameters[0].getHiddenUnitsPerRBM();
        for(int layer = 1; layer < layerParameters.length; layer++) {
            int currentLayerVisibleNodesIn = layerParameters[layer].getNumRBMS() * layerParameters[layer].getVisibleUnitsPerRBM();
            int currentLayerHiddenNodesIn = layerParameters[layer].getNumRBMS() * layerParameters[layer].getHiddenUnitsPerRBM();
            if(hiddenNodesOut != currentLayerVisibleNodesIn) {
                throw new IllegalArgumentException("Layer: " + layer + ", previous layer hidden nodes out (" + hiddenNodesOut + ") does not match current layer visible layers in (" + currentLayerVisibleNodesIn + ").");
            }

            //visibleNodesIn = currentLayerVisibleNodesIn;
            hiddenNodesOut = currentLayerHiddenNodesIn;
        }

    }

    public DeepRBM(final RBMLayer[] rbmLayers) {
        this.rbmLayers = rbmLayers;
        this.visibleSize = rbmLayers[0].size() * rbmLayers[0].getRBM(0).getVisibleSize();
        this.hiddenSize = rbmLayers[rbmLayers.length - 1].size() * rbmLayers[rbmLayers.length - 1].getRBM(0).getVisibleSize();
    }

    public RBMLayer[] getRbmLayers() {
        return rbmLayers;
    }

    public int getVisibleSize() {
        return visibleSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    @Override
    public String toString() {
        return "DeepRBM{" +
                "rbmLayers=" + PrettyPrint.toString(rbmLayers) +
                '}';
    }
}
