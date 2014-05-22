package nn.rbm.save;

import com.google.common.base.Function;
import math.matrix.Matrix;
import nn.rbm.RBM;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;

import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

/**
 * Created by kenny on 5/22/14.
 */
public class RBMPersister {

    private static final Logger LOGGER = Logger.getLogger(RBMPersister.class);

    private static final char DELIM = ',';

    public void save(final RBM rbm, final String file) {
        try {
            IOUtils.write(buildStringBuilderData(rbm).toString(), new FileOutputStream(file));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }
    }

    public StringBuilder buildStringBuilderData(final RBM rbm) {
        final StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(rbm.getVisibleSize()).append(DELIM).append(rbm.getHiddenSize()).append('\n');

        final Matrix weights = rbm.getWeights();
        for(int i = 0; i < rbm.getVisibleSize(); i++) {
            for(int j = 0; j < rbm.getHiddenSize(); j++) {
                stringBuilder.append(weights.get(i, j));
                if(j < rbm.getHiddenSize() - 1) {
                    stringBuilder.append(DELIM);
                }
            }
            stringBuilder.append('\n');
        }
        return stringBuilder;
    }

    public RBM load(final String file) {
        try {
            return buildRBM(IOUtils.readLines(new FileReader(file)));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
            return null;
        }
    }

    public RBM buildRBM(final List<String> lines) {
        final int[] metaData = COMMA_TO_INT_ARRAY_DESERIALIZER.apply(lines.get(0));
        final int visibleSize = metaData[0];
        final int hiddenSize = metaData[1];

        final RBM rbm = new RBM(visibleSize, hiddenSize);
        final Matrix weights = rbm.getWeights();
        for(int i = 0; i < visibleSize; i++) {
            double[] values = COMMA_TO_DOUBLE_ARRAY_DESERIALIZER.apply(lines.get(i + 1));
            for(int j = 0; j < hiddenSize; j++) {
                weights.set(i, j, values[j]);
            }
        }
        return rbm;
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

    private static final Function<String, double[]> COMMA_TO_DOUBLE_ARRAY_DESERIALIZER = new Function<String, double[]>() {
        @Override
        public double[] apply(String line) {
            String[] strValues = line.split(",");
            double[] doubleValues = new double[strValues.length];
            for(int i = 0; i < doubleValues.length; i++) {
                doubleValues[i] = Double.parseDouble(strValues[i]);
            }
            return doubleValues;
        }
    };

}
