package data.image;

import data.image.decode.Matrix24BitImageDecoder;
import data.image.encode.Matrix24BitImageEncoder;
import math.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/20/14.
 */
public class TestImage {

    @Test
    public void loadImage() {
        final Image image = new Image("/data/fighter_jet.jpg");
        assertEquals(400, image.width());
        assertEquals(250, image.height());
    }

    @Test
    public void encodeDecodeImage() {
        final Image image = new Image("/data/fighter_jet.jpg");
        //image.save("/tmp/fighter_save1.jpg");

        final Matrix24BitImageEncoder matrix24BitImageEncoder = new Matrix24BitImageEncoder();
        final Matrix matrix = matrix24BitImageEncoder.encode(image);
        assertEquals(400 * 24 * 250, matrix.cols());
        assertEquals(1, matrix.rows());

        final Matrix24BitImageDecoder matrix24BitImageDecoder = new Matrix24BitImageDecoder(250);
        final Image image2 = matrix24BitImageDecoder.decode(matrix);
        assertEquals(400, image2.width());
        assertEquals(250, image2.height());

        image2.save("/tmp/fighter_save.jpg");
    }
}
