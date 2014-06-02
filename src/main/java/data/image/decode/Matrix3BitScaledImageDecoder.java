package data.image.decode;

import data.image.Image;
import math.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class Matrix3BitScaledImageDecoder implements MatrixImageDecoder {

    private final int rows;

    private int cols = 1;

    public Matrix3BitScaledImageDecoder(int rows) {
        this.rows = rows;
    }

    @Override
    public Image decode(final Matrix matrix) {
        cols = matrix.columns() / 3 / rows;
        BufferedImage bi = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
        int y = 0;
        for(int x = 0; x < cols * rows; x++) {
            read(matrix, bi, x, y);
            if(x > 0 && (x % cols) == 0) {
                y++;
            }
        }
        return new Image(bi);
    }

    private void read(final Matrix matrix, final BufferedImage bi, final int x, final int y) {
        final int offset = x * 3;
        final int r = (int) (matrix.get(0, offset) * 255.0);
        final int g = (int) (matrix.get(0, offset + 1) * 255.0);
        final int b = (int) (matrix.get(0, offset + 2) * 255.0);
        final int rgb = (r << 16) + (g << 8) + b;
        bi.setRGB(x % cols, y, rgb);

    }

}
