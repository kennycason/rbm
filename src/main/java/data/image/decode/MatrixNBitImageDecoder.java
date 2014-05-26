package data.image.decode;

import data.image.Image;
import math.Matrix;

import java.awt.image.BufferedImage;

/**
 * Created by kenny on 5/20/14.
 */
public class MatrixNBitImageDecoder implements MatrixImageDecoder {

    private static final int RGB_BITS = 24;

    private final int bits;

    private static final int HIGH_BIT_FLAG = 0b100000000000000000000000;

    private final int rows;

    private int cols = 1;

    protected MatrixNBitImageDecoder(int bits, int rows) {
        this.bits = bits;
        this.rows = rows;
    }

    @Override
    public Image decode(final Matrix matrix) {
        cols = matrix.columns() / bits / rows;
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
            boolean set = matrix.get(0, ((x * bits) + offset)) > 0.5; // todo add a threshold variable
            if(set) {
                rgb += flag;
            }
            offset++;
            flag >>= RGB_BITS / bits;
        }
        bi.setRGB(x % cols, y, rgb);
    }

}
