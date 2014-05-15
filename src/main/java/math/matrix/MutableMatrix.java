package math.matrix;

import com.google.common.base.Function;

/**
 * Created by kenny on 5/13/14.
 */
public class MutableMatrix extends Matrix {

    public MutableMatrix(final int rows, final int cols) {
        super(rows, cols);
    }

    public MutableMatrix(final double[][] m) {
        super(m);
    }

    public MutableMatrix(final Matrix m) {
        super(m);
    }

    @Override
    public double[][] data() {
        return m;
    }

    @Override
    public Matrix set(final int i, final int j, final double value) {
        this.m[i][j] = value;
        return this;
    }

    @Override
    public Matrix dot(final Matrix m2) {
        if(cols != m2.rows) { throw new IllegalArgumentException("Matrices must be in form of A_n_m and B_m_p"); }

        final double[][] product = new double[rows][m2.cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < m2.cols; j++) {
                for(int k = 0; k < cols; k++) {
                    product[i][j] += m[i][k] * m2.get(k,j);
                }
            }
        }
        copy(product, m);
        return this;
    }

    @Override
    public Matrix add(final Matrix m2) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] += m2.get(i,j);
            }
        }
        return this;
    }

    @Override
    public Matrix subtract(final Matrix m2) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] -= m2.get(i,j);
            }
        }
        return this;
    }

    @Override
    public Matrix multiply(final double s) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] *= s;
            }
        }
        return this;
    }

    @Override
    public Matrix divide(final double s) {
        if(s == 0.0) { throw new IllegalArgumentException("Can not divide by zero!"); }
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] /= s;
            }
        }
        return this;
    }

    @Override
    public Matrix pow(final double p) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] = Math.pow(m[i][j], p);
            }
        }
        return this;
    }

    @Override
    public Matrix apply(final Function<Double, Double> f) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] = f.apply(m[i][j]);
            }
        }
        return this;
    }

    @Override
    public Matrix transpose() {
        final double[][] t = new double[cols][rows];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                t[j][i] = m[i][j];
            }
        }
        copy(t, m);
        return this;
    }

    @Override
    public Matrix fill(double value) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                m[i][j] = value;
            }
        }
        return this;
    }

    /*
        Static Helpers
     */
    public static MutableMatrix random(int r, int c) {
        return random(r, c, 1.0);
    }

    public static MutableMatrix random(int r, int c, double scalar) {
        final double[][] m = new double[r][c];
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                m[i][j] = RANDOM.nextDouble() * scalar;
            }
        }
        return new MutableMatrix(m);
    }

    public static MutableMatrix randomNormal(int r, int c) {
        return random(r, c, 1.0);
    }

    public static MutableMatrix randomNormal(int r, int c, double scalar) {
        final double[][] m = new double[r][c];
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                m[i][j] = RANDOM.nextGaussian() * scalar;
            }
        }
        return new MutableMatrix(m);
    }

}
