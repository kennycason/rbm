package utils;

import java.util.Arrays;

/**
 * Created by kenny on 5/13/14.
 */
public class PrettyPrint {

    private PrettyPrint() {}

    public static String toString(double[][] arrays) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("[");
        for(double[] array : arrays) {
            stringBuilder.append(Arrays.toString(array));
        }
        stringBuilder.append("]");
        return stringBuilder.toString();
    }

}
