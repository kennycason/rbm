package nn.rbm.deep;

import nn.rbm.RBM;
import nn.rbm.factory.RBMFactory;
import utils.PrettyPrint;

/**
 * Created by kenny on 5/16/14.
 */
public class DeepRBM {

    private final RBMLayer[] rbmLayers;


    public DeepRBM(final LayerParameters[] layerParameters, final RBMFactory rbmFactory) {
        rbmLayers = new RBMLayer[layerParameters.length];


        for(int layer = 0; layer < layerParameters.length; layer++) {
            final LayerParameters layerParameter = layerParameters[layer];
            final RBM[] rbms = new RBM[layerParameter.getNumRBMS()];

            for(int i = 0; i < layerParameter.getNumRBMS(); i++) {
                rbms[i] = rbmFactory.build(layerParameter.getVisibleUnitsPerRBM(), layerParameter.getHiddenUnitsPerRBM());
            }
            rbmLayers[layer] = new RBMLayer(rbms);
        }
    }

    public DeepRBM(final RBMLayer[] rbmLayers) {
        this.rbmLayers = rbmLayers;
    }

    public RBMLayer[] getRbmLayers() {
        return rbmLayers;
    }

    @Override
    public String toString() {
        return "DeepRBM{" +
                "rbmLayers=" + PrettyPrint.toString(rbmLayers) +
                '}';
    }
}
