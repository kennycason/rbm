package data.image.decode;

import data.image.Image;
import math.matrix.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class Matrix24BitImageDecoder implements MatrixImageDecoder {

    private static final int HIGH_BIT_FLAG = 0b100000000000000000000000;

    private int rows = 1;

    private int cols = 1;

    public Matrix24BitImageDecoder(int rows) {
        this.rows = rows;
    }

    @Override
    public Image decode(final Matrix matrix) {
        cols = matrix.cols() / 24 / rows;
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
        int rgb = 0;
        int offset = 0;
        int flag = HIGH_BIT_FLAG;
        while(flag > 0) {
            boolean set = matrix.get(0, ((x * 24) + offset)) > 0.5; // todo add a threshold variable
            if(set) {
                rgb += flag;
            }
            offset++;
            flag >>= 1;
        }
        bi.setRGB(x % cols, y, rgb);
    }

}
