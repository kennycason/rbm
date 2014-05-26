package math.matrix;

import math.DenseMatrix;
import math.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by kenny on 5/24/14.
 */
public class TestDenseMatrix {

    private static final double DELTA = 0.0;

    @Test
    public void dot() {
        // n * m
        Matrix a = DenseMatrix.make(new double[][]{{1, 2, 3}, {4, 5, 6}});
        // m * p
        Matrix b = DenseMatrix.make(new double[][]{{7, 8}, {9, 10}, {11, 12}});

        // should be [[58 64][139 154]]
        Matrix d = a.dot(b);

        assertEquals(2, a.rows());
        assertEquals(3, a.columns());

        assertEquals(58, d.get(0, 0), DELTA);
        assertEquals(64, d.get(0,1), DELTA);
        assertEquals(139, d.get(1,0), DELTA);
        assertEquals(154, d.get(1,1), DELTA);
    }

    @Test
    public void transpose() {
        Matrix m = DenseMatrix.make(new double[][]{{1, 0, 0}, {1, 1, 0}, {1, 1, 1}});

        // [[1.0, 1.0, 1.0][0.0, 1.0, 1.0][0.0, 0.0, 1.0]]
        final double[][] t = m.transpose().toArray();

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
        Matrix m = DenseMatrix.make(new double[][] {{1,0,0},{1,1,0},{1,1,1}});
        assertEquals(6, m.sum(), DELTA);
    }

    @Test
    public void pow() {
        Matrix m = DenseMatrix.make(new double[][]{{1,2},{3,4}});
        Matrix p = m.pow(2.0);

        assertEquals(1, m.get(0, 0), DELTA);
        assertEquals(4, m.get(0,1), DELTA);
        assertEquals(9, m.get(1,0), DELTA);
        assertEquals(16, m.get(1,1), DELTA);

        assertEquals(1, p.get(0,0), DELTA);
        assertEquals(4, p.get(0,1), DELTA);
        assertEquals(9, p.get(1,0), DELTA);
        assertEquals(16, p.get(1,1), DELTA);
    }

    @Test
    public void add() {
        Matrix m = DenseMatrix.make(new double[][]{{1,2},{3,4}});
        Matrix m2 = DenseMatrix.make(new double[][]{{1,2},{3,4}});
        Matrix p = m.add(m2);
        assertEquals(2, m.get(0,0), DELTA);
        assertEquals(4, m.get(0,1), DELTA);
        assertEquals(6, m.get(1,0), DELTA);
        assertEquals(8, m.get(1,1), DELTA);

        assertEquals(2, p.get(0,0), DELTA);
        assertEquals(4, p.get(0,1), DELTA);
        assertEquals(6, p.get(1,0), DELTA);
        assertEquals(8, p.get(1,1), DELTA);
    }

    @Test
    public void subtract() {
        Matrix m = DenseMatrix.make(new double[][]{{1,2},{3,4}});
        Matrix m2 = DenseMatrix.make(new double[][]{{1,2},{3,4}});
        Matrix p = m.subtract(m2);
        assertEquals(0, m.get(0,0), DELTA);
        assertEquals(0, m.get(0,1), DELTA);
        assertEquals(0, m.get(1,0), DELTA);
        assertEquals(0, m.get(1,1), DELTA);

        assertEquals(0, p.get(0,0), DELTA);
        assertEquals(0, p.get(0,1), DELTA);
        assertEquals(0, p.get(1,0), DELTA);
        assertEquals(0, p.get(1,1), DELTA);
    }

    @Test
    public void divide() {
        Matrix m = DenseMatrix.make(new double[][]{{2, 4}, {6, 8}});
        double[][] d = m.divide(2).toArray();
        assertEquals(1, d[0][0], DELTA);
        assertEquals(2, d[0][1], DELTA);
        assertEquals(3, d[1][0], DELTA);
        assertEquals(4, d[1][1], DELTA);
    }

    @Test
    public void addColumns() {
        Matrix a = DenseMatrix.make(new double[][]{{1, 2}, {3, 4}});
        Matrix b = DenseMatrix.make(new double[][]{{5, 6}, {7, 8}});
        Matrix m = a.addColumns(b);

        assertEquals(2, a.rows());
        assertEquals(2, a.columns());
        assertEquals(1, a.get(0,0), DELTA);
        assertEquals(2, a.get(0,1), DELTA);
        assertEquals(3, a.get(1,0), DELTA);
        assertEquals(4, a.get(1,1), DELTA);

        assertEquals(2, m.rows());
        assertEquals(4, m.columns());
        assertEquals(1, m.get(0,0), DELTA);
        assertEquals(2, m.get(0,1), DELTA);
        assertEquals(3, m.get(1,0), DELTA);
        assertEquals(4, m.get(1,1), DELTA);

        assertEquals(5, m.get(0,2), DELTA);
        assertEquals(6, m.get(0,3), DELTA);
        assertEquals(7, m.get(1,2), DELTA);
        assertEquals(8, m.get(1,3), DELTA);
    }

}
