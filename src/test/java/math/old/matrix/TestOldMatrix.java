package math.old.matrix;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/16/14.
 */
public class TestOldMatrix {

    private static final double DELTA = 0.0;

    @Test
    public void appendRows() {
        double[][] a = new double[][] {{1,1,1}};
        double[][] b = new double[][] {{2,2,2}};
        double[][] c = new double[][] {{3,3,3}};

        double[][] m = Matrix.appendRows(a, b, c);
        assertEquals(3, Matrix.rows(m));
        assertEquals(3, Matrix.cols(m));

        assertEquals(1, m[0][0], DELTA);
        assertEquals(1, m[0][1], DELTA);
        assertEquals(1, m[0][2], DELTA);
        assertEquals(2, m[1][0], DELTA);
        assertEquals(2, m[1][1], DELTA);
        assertEquals(2, m[1][2], DELTA);
        assertEquals(3, m[2][0], DELTA);
        assertEquals(3, m[2][1], DELTA);
        assertEquals(3, m[2][2], DELTA);
    }

    @Test
    public void appendColumns() {
        double[][] a = new double[][] {{1,1,1}};
        double[][] b = new double[][] {{2,2,2}};
        double[][] c = new double[][] {{3,3,3}};

        double[][] m = Matrix.appendColumns(a, b, c);
        assertEquals(1, Matrix.rows(m));
        assertEquals(9, Matrix.cols(m));

        assertEquals(1, m[0][0], DELTA);
        assertEquals(1, m[0][1], DELTA);
        assertEquals(1, m[0][2], DELTA);
        assertEquals(2, m[0][3], DELTA);
        assertEquals(2, m[0][4], DELTA);
        assertEquals(2, m[0][5], DELTA);
        assertEquals(3, m[0][6], DELTA);
        assertEquals(3, m[0][7], DELTA);
        assertEquals(3, m[0][8], DELTA);
    }

    @Test
    public void appendColumns2() {
        double[][] a = new double[][] {{1},{2}};

        double[][] m = Matrix.appendColumns(a, a, a);
        assertEquals(2, Matrix.rows(m));
        assertEquals(3, Matrix.cols(m));

        assertEquals(1, m[0][0], DELTA);
        assertEquals(2, m[1][0], DELTA);
        assertEquals(1, m[0][1], DELTA);
        assertEquals(2, m[1][1], DELTA);
        assertEquals(1, m[0][2], DELTA);
        assertEquals(2, m[1][2], DELTA);
    }

    @Test
    public void splitColumns() {
        double[][] a = new double[][] {{1,1,1},{2,2,2},{3,3,3}};

        List<double[][]> m = Matrix.splitColumns(a, 3);
        assertEquals(3, Matrix.rows(m.get(0)));
        assertEquals(1, Matrix.cols(m.get(0)));

        for(int i = 0; i < 3; i++) {
            assertEquals(1, m.get(i)[0][0], DELTA);
            assertEquals(2, m.get(i)[1][0], DELTA);
            assertEquals(3, m.get(i)[2][0], DELTA);
        }
    }

}
