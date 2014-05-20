package data.image.decode;

import data.image.Image;
import math.matrix.Matrix;

public interface MatrixImageDecoder {

    Image decode(final Matrix matrix);

}
