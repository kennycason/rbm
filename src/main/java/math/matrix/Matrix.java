package math.matrix;

import com.google.common.base.Function;
import utils.PrettyPrint;

import java.util.ArrayList;
import java.util.List;
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

    public abstract double[] row(final int row);

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

    public abstract Matrix fill(final double value);

    public abstract Matrix appendRows(final Matrix m2);

    public abstract List<Matrix> splitColumns(final int pieces);

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

    public static double[][] appendRows(final double[][]... m) {
        int rows = 0;
        for(int i = 0; i < m.length; i++) {
            rows += rows(m[0]);
        }
        final int cols = cols(m[0]);
        final double[][] appended = new double[rows][cols];

        int offset = 0;
        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[i].length; j++) {
                System.arraycopy(m[i][j], 0, appended[offset], 0, cols);
                offset++;
            }
        }
        return appended;
    }

    public static double[][] appendColumns(final double[][]... m) {
        int totalCols = 0;
        for(int i = 0; i < m.length; i++) {
            totalCols += cols(m[i]);
        }
        final int rows = rows(m[0]);
        final double[][] appended = new double[rows][totalCols];
        for(int k = 0; k < m.length; k++) {
            final int cols = cols(m[k]);
            for(int i = 0; i < rows; i++) {
                final int start = k * cols;
                System.arraycopy(m[k][i], 0, appended[i], start, cols);
            }
        }
        return appended;
    }

    public static List<double[][]> splitColumns(final double[][] m, int numPieces) {
        List<double[][]> pieces = new ArrayList<>(numPieces);
        final int rows = rows(m);
        final int cols = cols(m) / numPieces; // must be evenly splittable
        for(int p = 0; p < numPieces; p++) {
            final double[][] piece = new double[rows][cols];
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    piece[i][j] = m[i][j + (p * cols)];
                }
            }
            pieces.add(piece);
        }
        return pieces;
    }

    public static int rows(final double[][] m) {
        return m.length;
    }

    public static int cols(final double[][] m) {
        return m[0].length;
    }

}
