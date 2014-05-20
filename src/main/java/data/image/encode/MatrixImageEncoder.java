package data.image.encode;

import data.image.Image;
import math.matrix.Matrix;

/**
 * Created by kenny on 5/20/14.
 */
public interface MatrixImageEncoder {

    Matrix encode(final Image image);

}
