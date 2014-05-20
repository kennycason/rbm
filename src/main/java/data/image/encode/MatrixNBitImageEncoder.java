package data.image.encode;

import data.image.Image;
import math.matrix.ImmutableMatrix;
import math.matrix.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class MatrixNBitImageEncoder implements MatrixImageEncoder {

    private static final int RGB_BITS = 24;

    private final int bits;

    private static final int HIGH_BIT_FLAG = 0b1000_0000_0000_0000_0000_0000;

    protected MatrixNBitImageEncoder(int bits) {
        this.bits = bits;
    }

    @Override
    public Matrix encode(final Image image) {
        double[] matrix = new double[image.width() * bits * image.height()];
        final BufferedImage bi = image.data();
        for(int y = 0; y < image.height(); y++) {
            for(int x = 0; x < image.width(); x++) {
                read(matrix, bi, x, y);
            }
        }
        return new ImmutableMatrix(new double[][] { matrix });
    }

    private void read(final double[] matrix, final BufferedImage bi, final int x, final int y) {
        int flag = HIGH_BIT_FLAG;
        int offset = 0;
        while(flag > 0) {
            final int rgb = bi.getRGB(x, y) & 0xFFFFFF;
            boolean set = (rgb & flag) == flag;
            int index = (y * bi.getWidth() * bits) + ((x * bits) + offset);
            matrix[index] = set ? 1.0 : 0.0;
            offset++;
            flag >>= RGB_BITS / bits;
        }
    }
}
