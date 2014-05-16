package math.matrix;

import org.apache.log4j.Logger;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/13/14.
 */
public class TestImmutableMatrix {

    private static final Logger LOGGER = Logger.getLogger(TestImmutableMatrix.class);

    private static final double DELTA = 0.0;

    @Test
    public void dot() {
        // n * m
        final double[][] ad = new double[][]{{1,2,3},{4,5,6}};
        ImmutableMatrix a = new ImmutableMatrix(ad);

        // m * p
        final double[][] bd = new double[][]{{7,8},{9,10},{11,12}};
        ImmutableMatrix b = new ImmutableMatrix(bd);

        // should be [[58 64][139 154]]
        Matrix p = a.dot(b);

        assertEquals(58, p.get(0,0), DELTA);
        assertEquals(64, p.get(0,1), DELTA);
        assertEquals(139, p.get(1,0), DELTA);
        assertEquals(154, p.get(1,1), DELTA);

        // assert unchanged
        assertEquals(1, ad[0][0], DELTA);
        assertEquals(2, ad[0][1], DELTA);
        assertEquals(3, ad[0][2], DELTA);
        assertEquals(4, ad[1][0], DELTA);
        assertEquals(5, ad[1][1], DELTA);
        assertEquals(6, ad[1][2], DELTA);

        assertEquals(7, bd[0][0], DELTA);
        assertEquals(8, bd[0][1], DELTA);
        assertEquals(9, bd[1][0], DELTA);
        assertEquals(10, bd[1][1], DELTA);
        assertEquals(11, bd[2][0], DELTA);
        assertEquals(12, bd[2][1], DELTA);
    }

    @Test
    public void transpose() {
        final double[][] md = new double[][]{{1,0,0},{1,1,0},{1,1,1}};
        ImmutableMatrix m = new ImmutableMatrix(md);

        // [[1.0, 1.0, 1.0][0.0, 1.0, 1.0][0.0, 0.0, 1.0]]
        final double[][] t = m.transpose().data();
        assertEquals(1, t[0][0], DELTA);
        assertEquals(1, t[0][1], DELTA);
        assertEquals(1, t[0][2], DELTA);
        assertEquals(0, t[1][0], DELTA);
        assertEquals(1, t[1][1], DELTA);
        assertEquals(1, t[1][2], DELTA);
        assertEquals(0, t[2][0], DELTA);
        assertEquals(0, t[2][1], DELTA);
        assertEquals(1, t[2][2], DELTA);

        // unchanged
        assertEquals(1, md[0][0], DELTA);
        assertEquals(0, md[0][1], DELTA);
        assertEquals(0, md[0][2], DELTA);
        assertEquals(1, md[1][0], DELTA);
        assertEquals(1, md[1][1], DELTA);
        assertEquals(0, md[1][2], DELTA);
        assertEquals(1, md[2][0], DELTA);
        assertEquals(1, md[2][1], DELTA);
        assertEquals(1, md[2][2], DELTA);
    }

    @Test
    public void sum() {
        ImmutableMatrix m = new ImmutableMatrix(new double[][] {{1,0,0},{1,1,0},{1,1,1}});
        assertEquals(6, m.sum(), DELTA);
    }

    @Test
    public void pow() {
        ImmutableMatrix m = new ImmutableMatrix(new double[][]{{1,2},{3,4}});
        double[][] pow = m.pow(2).data();
        assertEquals(1, pow[0][0], DELTA);
        assertEquals(4, pow[0][1], DELTA);
        assertEquals(9, pow[1][0], DELTA);
        assertEquals(16, pow[1][1], DELTA);
    }

    @Test
    public void divide() {
        ImmutableMatrix m = new ImmutableMatrix(new double[][]{{2,4},{6,8}});
        double[][] d = m.divide(2).data();
        assertEquals(1, d[0][0], DELTA);
        assertEquals(2, d[0][1], DELTA);
        assertEquals(3, d[1][0], DELTA);
        assertEquals(4, d[1][1], DELTA);
    }

    @Test
    public void add() {
        ImmutableMatrix m = new ImmutableMatrix(new double[][]{{2,4},{6,8}});
        ImmutableMatrix m2 = new ImmutableMatrix(new double[][]{{2,4},{6,8}});
        double[][] d = m.add(m2).data();
        assertEquals(4, d[0][0], DELTA);
        assertEquals(8, d[0][1], DELTA);
        assertEquals(12, d[1][0], DELTA);
        assertEquals(16, d[1][1], DELTA);
    }

    @Test
    public void subtract() {
        ImmutableMatrix m = new ImmutableMatrix(new double[][]{{2,4},{6,8}});
        ImmutableMatrix m2 = new ImmutableMatrix(new double[][]{{2,4},{6,8}});
        double[][] d = m.subtract(m2).data();
        assertEquals(0, d[0][0], DELTA);
        assertEquals(0, d[0][1], DELTA);
        assertEquals(0, d[1][0], DELTA);
        assertEquals(0, d[1][1], DELTA);
    }

    @Test(expected = IllegalArgumentException.class)
    public void dotFail() {
        // n * m
        ImmutableMatrix a = new ImmutableMatrix(new double[][]
                        {{1,1,1,1},
                        {2,2,2,2},
                        {3,3,3,3}});

        // q * p
        ImmutableMatrix b = new ImmutableMatrix(new double[][]
                        {{1,1,1},
                        {2,2,2},
                        {3,3,3}});

        LOGGER.info(a.dot(b));
    }

}
