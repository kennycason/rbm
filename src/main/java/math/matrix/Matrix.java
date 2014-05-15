package math.matrix;

import com.google.common.base.Function;
import utils.PrettyPrint;

import java.util.Random;

/**
 * Created by kenny on 5/14/14.
 */
public abstract class Matrix {

    protected static final Random RANDOM = new Random();

    protected final int rows;

    protected final int cols;

    protected final int dim;

    protected final double[][] m;

    protected Matrix(final int rows, final int cols) {
        this.rows = rows;
        this.cols = cols;
        this.dim = rows * cols;
        this.m = new double[rows][cols];
    }

    protected Matrix(final double[][] m) {
        this.rows = rows(m);
        this.cols = cols(m);
        this.dim = this.rows * this.cols;
        this.m = new double[this.rows][this.cols];
        copy(m, this.m);
    }

    protected Matrix(Matrix m) {
        this.rows = m.rows();
        this.cols = m.cols();
        this.dim = m.dim();
        this.m = copy(m.data());
    }

    public abstract double[][] data();

    public abstract Matrix set(final int i, final int j, final double value);

    public int dim() {
        return dim;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public double get(final int i, final int j) {
        return m[i][j];
    }

    public abstract Matrix dot(final Matrix m2);

    public abstract Matrix add(final Matrix m2);

    public abstract Matrix subtract(final Matrix m2);

    public abstract Matrix multiply(final double s);

    public abstract Matrix divide(final double s);

    public abstract Matrix pow(final double s);

    public double sum() {
        double sum = 0.0;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                sum += m[i][j];
            }
        }
        return sum;
    }

    public abstract Matrix apply(final Function<Double, Double> f);

    public abstract Matrix transpose();

    public abstract Matrix fill(double value);

    @Override
    public String toString() {
        return PrettyPrint.toString(m);
    }

    /*
        Static Helpers
     */
    public static double[][] copy(final double[][] m) {
        final int rows = rows(m);
        final int cols = cols(m);
        final double[][] c = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                c[i][j] = m[i][j];
            }
        }
        return c;
    }

    public static void copy(final double[][] from, final double[][] to) {
        final int rows = rows(from);
        final int cols = cols(from);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                to[i][j] = from[i][j];
            }
        }
    }

    public static int rows(final double[][] m) {
        return m.length;
    }

    public static int cols(final double[][] m) {
        return m[0].length;
    }

}
