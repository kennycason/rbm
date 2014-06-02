package data.image.encode;

import data.image.Image;
import math.DenseMatrix;
import math.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class Matrix3BitScaledImageEncoder implements MatrixImageEncoder {

    private int ignoreRGB = -1;

    public Matrix3BitScaledImageEncoder() {}

    public Matrix3BitScaledImageEncoder(final int ignoreRGB) {
        this.ignoreRGB = ignoreRGB;
    }

    @Override
    public Matrix encode(final Image image) {
        double[] matrix = new double[image.width() * 3 * image.height()];
        final BufferedImage bi = image.data();
        for(int y = 0; y < image.height(); y++) {
            for(int x = 0; x < image.width(); x++) {
                read(matrix, bi, x, y);
            }
        }
        return DenseMatrix.make(new double[][]{matrix});
    }

    private void read(final double[] matrix, final BufferedImage bi, final int x, final int y) {
        final int rgb = bi.getRGB(x, y) & 0xFFFFFF;
        if(rgb == ignoreRGB) { return; }

        final int r = (rgb & 0xFF0000) >> 16;
        final int g = (rgb & 0xFF00) >> 8;
        final int b = (rgb & 0xFF);

        int index = (y * bi.getWidth() * 3) + (x * 3);
        matrix[index] = r / 255.0;
        matrix[index + 1] = g / 255.0;
        matrix[index + 2] = b / 255.0;
    }

}
