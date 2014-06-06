package nn.rbm.save;

import com.google.common.base.Function;
import nn.rbm.RBM;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.deep.RBMLayer;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * Created by kenny on 5/22/14.
 */
public class DeepRBMPersister {

    private static final Logger LOGGER = Logger.getLogger(DeepRBMPersister.class);

    private static final char DELIM = ',';

    private static final RBMPersister RBM_PERSISTER = new RBMPersister();

    public void save(final DeepRBM deepRBM, final String file) {
        try {
            LOGGER.info("Saving Deep RBM to " + file);
            FileWriter writer = new FileWriter(file);

            // write out layer info
            final RBMLayer[] rbmLayers = deepRBM.getRbmLayers();
            for(int l = 0; l < rbmLayers.length; l++) {
                writer.write(String.valueOf(rbmLayers[l].size()));
                writer.write(DELIM);
                writer.write(String.valueOf(rbmLayers[l].getRBM(0).getVisibleSize()));
                writer.write(DELIM);
                writer.write(String.valueOf(rbmLayers[l].getRBM(0).getHiddenSize()));
                if(l < rbmLayers.length - 1) {
                    writer.write(DELIM);
                }
            }
            writer.write('\n');

            // for each layer, write out each rbm
            for(int l = 0; l < rbmLayers.length; l++) {
                for(int r = 0; r < rbmLayers[l].size(); r++) {
                    RBM_PERSISTER.writeStringBuilderData(rbmLayers[l].getRBM(r), writer);
                }
            }
            writer.close();
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }
    }

    public DeepRBM load(final String file) {
        try {
            List<String> lines = IOUtils.readLines(new FileReader(file));

            final int[] layerInfo = COMMA_TO_INT_ARRAY_DESERIALIZER.apply(lines.get(0));
            final int layers = layerInfo.length / 3;

            final LayerParameters[] layerParameters = new LayerParameters[layers];
            for(int l = 0; l < layers; l++) {
                layerParameters[l] = new LayerParameters().setNumRBMS(layerInfo[l * 3]).setVisibleUnitsPerRBM(layerInfo[l * 3 + 1]).setHiddenUnitsPerRBM(layerInfo[l * 3 + 2]);
            }

            final RBMLayer[] rbmLayers = new RBMLayer[layers];

            int startIndex = 1;
            for(int l = 0; l < layers; l++) {
                final RBM[] rbms = new RBM[layerParameters[l].getNumRBMS()];
                final int length = 1 + layerParameters[l].getVisibleUnitsPerRBM();
                for(int r = 0; r < layerParameters[l].getNumRBMS(); r++) {
                    final List<String> rbmData = lines.subList(startIndex, startIndex + length);
                    rbms[r] = RBM_PERSISTER.buildRBM(rbmData);
                    startIndex += length;
                }
                rbmLayers[l] = new RBMLayer(rbms);
            }

            return new DeepRBM(rbmLayers);
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
            return null;
        }
    }

    private static final Function<String, int[]> COMMA_TO_INT_ARRAY_DESERIALIZER = new Function<String, int[]>() {
        @Override
        public int[] apply(String line) {
            String[] strValues = line.split(",");
            int[] intValues = new int[strValues.length];
            for(int i = 0; i < intValues.length; i++) {
                intValues[i] = Integer.parseInt(strValues[i]);
            }
            return intValues;
        }
    };

}
