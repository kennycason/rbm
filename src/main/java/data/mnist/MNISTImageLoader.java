package data.mnist;


import math.DenseMatrix;
import math.Matrix;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Created by kenny on 5/15/14.
 *
 * http://yann.lecun.com/exdb/mnist/
 */
public class MNISTImageLoader {

    private static final Logger LOGGER = Logger.getLogger(MNISTImageLoader.class);

    /*
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel

        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
     */
    public Matrix loadIdx3(String file) {
        try {
            final ByteBuffer byteBuffer = ByteBuffer.wrap(IOUtils.toByteArray(MNISTImageLoader.class.getResourceAsStream(file)));
            final int magicNumber = byteBuffer.getInt(); // 2051
            final int numberImages = byteBuffer.getInt();
            final int numberRows = byteBuffer.getInt();
            final int numberCols = byteBuffer.getInt();

            final double[][] data = new double[numberImages][];
            for(int i = 0; i < numberImages; i++) {
                data[i] = readImage(byteBuffer, numberRows, numberCols);
            }
            return DenseMatrix.make(data);
        } catch(IOException e) {
            LOGGER.error(e);
            return DenseMatrix.make(new double[0][0]);
        }
    }

    private double[] readImage(ByteBuffer byteBuffer, int numberRows, int numberCols) {
        final double[] data = new double[numberRows * numberCols];
        for(int i = 0; i < numberRows; i++) {
            for(int j = 0; j < numberCols; j++) {
                data[(i * numberCols) + j] = byteBuffer.get() & 0xFF;
            }
        }
        return data;
    }

    /*
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  10000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label

        The labels values are 0 to 9.
    */
    public Matrix loadIdx1(String file) {
        try {
            final ByteBuffer byteBuffer = ByteBuffer.wrap(IOUtils.toByteArray(MNISTImageLoader.class.getResourceAsStream(file)));
            final int magicNumber = byteBuffer.getInt(); // 2049
            final int numberItems = byteBuffer.getInt();

            final double[][] data = new double[numberItems][];
            for(int i = 0; i < numberItems; i++) {
                data[i] = new double[] { byteBuffer.get() & 0xFF };
            }
            return DenseMatrix.make(data);
        } catch(IOException e) {
            LOGGER.error(e);
            return DenseMatrix.make(new double[0][0]);
        }
    }

}
