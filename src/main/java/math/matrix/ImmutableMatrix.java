package math.matrix;

import com.google.common.base.Function;


/**
 * Created by kenny on 5/14/14.
 */
public class ImmutableMatrix extends Matrix {

    public ImmutableMatrix(final int rows, final int cols) {
        super(rows, cols);
    }

    public ImmutableMatrix(final double[][] m) {
        super(m);
    }

    public ImmutableMatrix(final Matrix m) {
        super(m);
    }

    @Override
    public double[][] data() {
        return copy(m);
    }

    @Override
    public Matrix set(final int i, final int j, final double value) {
        double[][] copy = copy(this.m);
        copy[i][j] = j;
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix dot(final Matrix m2) {
        return dot(this, m2);
    }

    public static ImmutableMatrix dot(final Matrix m1, final Matrix m2) {
        if(m1.cols != m2.rows) { throw new IllegalArgumentException("Matrices must be in form of A_n_m and B_m_p"); }

        final double[][] product = new double[m1.rows][m2.cols];
        for(int i = 0; i < m1.rows; i++) {
            for(int j = 0; j < m2.cols; j++) {
                for(int k = 0; k < m1.cols; k++) {
                    product[i][j] += m1.get(i, k) * m2.get(k,j);
                }
            }
        }
        return new ImmutableMatrix(product);
    }

    @Override
    public ImmutableMatrix add(final Matrix m2) {
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = m[i][j] + m2.get(i,j);
            }
        }
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix subtract(final Matrix m2) {
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = m[i][j] - m2.get(i,j);
            }
        }
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix multiply(final double s) {
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = m[i][j] * s;
            }
        }
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix divide(final double s) {
        if(s == 0.0) { throw new IllegalArgumentException("Can not divide by zero!"); }
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = m[i][j] / s;
            }
        }
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix pow(final double p) {
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = Math.pow(m[i][j], p);
            }
        }
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix apply(final Function<Double, Double> f) {
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = f.apply(m[i][j]);
            }
        }
        return new ImmutableMatrix(copy);
    }

    @Override
    public ImmutableMatrix transpose() {
        return transpose(this);
    }

    public static ImmutableMatrix transpose(Matrix m) {
        final double[][] t = new double[m.cols][m.rows];
        for(int i = 0; i < m.rows; i++) {
            for(int j = 0; j < m.cols; j++) {
                t[j][i] = m.get(i, j);
            }
        }
        return new ImmutableMatrix(t);
    }

    @Override
    public ImmutableMatrix fill(double value) {
        final double[][] copy = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                copy[i][j] = value;
            }
        }
        return new ImmutableMatrix(copy);
    }

    /*
        Static Helpers
     */
    public static ImmutableMatrix random(int r, int c) {
        return random(r, c, 1.0);
    }

    public static ImmutableMatrix random(int r, int c, double scalar) {
        final double[][] m = new double[r][c];
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                m[i][j] = RANDOM.nextDouble() * scalar;
            }
        }
        return new ImmutableMatrix(m);
    }

    public static ImmutableMatrix randomNormal(int r, int c) {
        return random(r, c, 1.0);
    }

    public static ImmutableMatrix randomNormal(int r, int c, double scalar) {
        final double[][] m = new double[r][c];
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                m[i][j] = RANDOM.nextGaussian() * scalar;
            }
        }
        return new ImmutableMatrix(m);
    }

}
