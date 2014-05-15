package math.matrix;

import org.apache.log4j.Logger;
import org.junit.Test;
import utils.PrettyPrint;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/13/14.
 */
public class TestMutableMatrix {

    private static final Logger LOGGER = Logger.getLogger(TestMutableMatrix.class);

    private static final double DELTA = 0.0;

    @Test
    public void dot() {
        // n * m
        MutableMatrix a = new MutableMatrix(new double[][]{{1,2,3},{4,5,6}});

        // m * p
        MutableMatrix b = new MutableMatrix(new double[][]{{7,8},{9,10},{11,12}});

        // should be [[58 64][139 154]]
        a.dot(b);

        assertEquals(58, a.get(0,0), DELTA);
        assertEquals(64, a.get(0,1), DELTA);
        assertEquals(139, a.get(1,0), DELTA);
        assertEquals(154, a.get(1,1), DELTA);
    }

    @Test
    public void transpose() {
        MutableMatrix m = new MutableMatrix(new double[][]{{1,0,0},{1,1,0},{1,1,1}});

        // [[1.0, 1.0, 1.0][0.0, 1.0, 1.0][0.0, 0.0, 1.0]]
        final double[][] t = m.transpose().data();
        LOGGER.info(PrettyPrint.toString(t));
        assertEquals(1, t[0][0], DELTA);
        assertEquals(1, t[0][1], DELTA);
        assertEquals(1, t[0][2], DELTA);
        assertEquals(0, t[1][0], DELTA);
        assertEquals(1, t[1][1], DELTA);
        assertEquals(1, t[1][2], DELTA);
        assertEquals(0, t[2][0], DELTA);
        assertEquals(0, t[2][1], DELTA);
        assertEquals(1, t[2][2], DELTA);
    }

    @Test
    public void sum() {
        MutableMatrix m = new MutableMatrix(new double[][] {{1,0,0},{1,1,0},{1,1,1}});
        assertEquals(6, m.sum(), DELTA);
    }

    @Test
    public void pow() {
        MutableMatrix m = new MutableMatrix(new double[][]{{1,2},{3,4}});
        m.pow(2);
        final double[][] d = m.data();
        assertEquals(1, d[0][0], DELTA);
        assertEquals(4, d[0][1], DELTA);
        assertEquals(9, d[1][0], DELTA);
        assertEquals(16, d[1][1], DELTA);
    }

    @Test
    public void divide() {
        MutableMatrix m = new MutableMatrix(new double[][]{{2,4},{6,8}});
        m.divide(2);
        final double[][] d = m.data();
        assertEquals(1, d[0][0], DELTA);
        assertEquals(2, d[0][1], DELTA);
        assertEquals(3, d[1][0], DELTA);
        assertEquals(4, d[1][1], DELTA);
    }

    @Test
    public void add() {
        MutableMatrix m = new MutableMatrix(new double[][]{{2,4},{6,8}});
        MutableMatrix m2 = new MutableMatrix(new double[][]{{2,4},{6,8}});
        m.add(m2);
        final double[][] d = m.data();
        assertEquals(4, d[0][0], DELTA);
        assertEquals(8, d[0][1], DELTA);
        assertEquals(12, d[1][0], DELTA);
        assertEquals(16, d[1][1], DELTA);
    }

    @Test
    public void subtract() {
        MutableMatrix m = new MutableMatrix(new double[][]{{2,4},{6,8}});
        MutableMatrix m2 = new MutableMatrix(new double[][]{{2,4},{6,8}});
        m.subtract(m2);
        final double[][] d = m.data();
        assertEquals(0, d[0][0], DELTA);
        assertEquals(0, d[0][1], DELTA);
        assertEquals(0, d[1][0], DELTA);
        assertEquals(0, d[1][1], DELTA);
    }

    @Test(expected = IllegalArgumentException.class)
    public void dotFail() {
        // n * m
        MutableMatrix a = new MutableMatrix(new double[][]
                        {{1,1,1,1},
                        {2,2,2,2},
                        {3,3,3,3}});

        // q * p
        MutableMatrix b = new MutableMatrix(new double[][]
                        {{1,1,1},
                        {2,2,2},
                        {3,3,3}});

        LOGGER.info(a.dot(b));
    }

}
