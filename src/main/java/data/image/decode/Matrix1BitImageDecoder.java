package data.image.decode;

import data.image.Image;
import math.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class Matrix1BitImageDecoder implements MatrixImageDecoder {

    private int rows = 1;

    private int cols = 1;

    public Matrix1BitImageDecoder(int rows) {
        this.rows = rows;
    }

    @Override
    public Image decode(final Matrix matrix) {
        cols = matrix.columns() / rows;
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
        boolean set = matrix.get(0, x) > 0.5; // todo add a threshold variable
        if(set) {
            bi.setRGB(x % cols, y, 0x00);
        } else {
            bi.setRGB(x % cols, y, 0xFFFFFF);
        }
    }

}
