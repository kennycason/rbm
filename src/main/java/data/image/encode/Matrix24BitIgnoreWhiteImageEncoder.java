package data.image.encode;

import data.image.Image;
import math.DenseMatrix;
import math.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class Matrix24BitIgnoreWhiteImageEncoder implements MatrixImageEncoder {

    private static final int RGB_BITS = 24;

    private final int bits = 24;

    private static final int HIGH_BIT_FLAG = 0b1000_0000_0000_0000_0000_0000;

    public Matrix24BitIgnoreWhiteImageEncoder() {}

    @Override
    public Matrix encode(final Image image) {
        double[] matrix = new double[image.width() * bits * image.height()];
        final BufferedImage bi = image.data();
        for(int y = 0; y < image.height(); y++) {
            for(int x = 0; x < image.width(); x++) {
                read(matrix, bi, x, y);
            }
        }
        return DenseMatrix.make(new double[][]{matrix});
    }

    private void read(final double[] matrix, final BufferedImage bi, final int x, final int y) {
        int flag = HIGH_BIT_FLAG;
        int offset = 0;
        while(flag > 0) {
            int rgb = bi.getRGB(x, y) & 0xFFFFFF;
            if(rgb == 0xFFFFFF) {
                rgb = 0x000000;
            }
            boolean set = (rgb & flag) == flag;
            int index = (y * bi.getWidth() * bits) + ((x * bits) + offset);
            matrix[index] = set ? 1.0 : 0.0;
            offset++;
            flag >>= RGB_BITS / bits;
        }
    }
}
