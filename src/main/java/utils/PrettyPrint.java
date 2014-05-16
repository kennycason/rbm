package utils;

import java.util.Arrays;

/**
 * Created by kenny on 5/13/14.
 */
public class PrettyPrint {

    private PrettyPrint() {}

    public static String toString(double[][] arrays) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("[\n");
        for(double[] array : arrays) {
            stringBuilder.append(Arrays.toString(array));
            stringBuilder.append('\n');
        }
        stringBuilder.append("]");
        return stringBuilder.toString();
    }

    public static String toPixelBox(double[][] arrays, double threshold) {
        StringBuilder stringBuilder = new StringBuilder();
        for(double[] array : arrays) {
            for(int i = 0; i < array.length; i++) {
                if(array[i] >= threshold) {
                    stringBuilder.append("■");
                } else {
                    stringBuilder.append("□");
                }
            }
            stringBuilder.append('\n');
        }
        return stringBuilder.toString();
    }

    public static String toPixelBox(double[] array, int columnSize, double threshold) {
        final int rowSize = array.length / columnSize;
        final double[][] matrix = new double[rowSize][columnSize];
        for(int i = 0; i < rowSize; i++) {
            for(int j = 0; j < columnSize; j++) {
                matrix[i][j] = array[(i * columnSize) + j];
            }
        }
        return toPixelBox(matrix, threshold);
    }
}
