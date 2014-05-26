package math;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.SparseDoubleAlgebra;
import math.functions.Power;
import math.functions.RandomDouble;
import math.functions.RandomGaussian;
import math.functions.doubledouble.Add;
import math.functions.doubledouble.Divide;
import math.functions.doubledouble.Multiply;
import math.functions.doubledouble.Subtract;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/24/14.
 */
public abstract class Matrix {

    protected static final DoubleDoubleFunction ADD = new Add();
    protected static final DoubleDoubleFunction SUBTRACT = new Subtract();
    protected static final DoubleDoubleFunction MULTIPLY = new Multiply();
    protected static final DoubleDoubleFunction DIVIDE = new Divide();
    protected static final DoubleFunction RANDOM_GAUSSIAN = new RandomGaussian();
    protected static final DoubleFunction RANDOM_DOUBLE = new RandomDouble(1.0);
    protected static final DenseDoubleAlgebra DENSE_DOUBLE_ALGEBRA = new DenseDoubleAlgebra();
    protected static final SparseDoubleAlgebra SPARSE_DOUBLE_ALGEBRA = new SparseDoubleAlgebra();

    protected DoubleMatrix2D m;

    protected Matrix(DoubleMatrix2D m) {
        this.m = m;
    }

    /* IMMUTABLE OPERATIONS */

    public abstract Matrix copy();

    public abstract Matrix transpose();

    public abstract Matrix dot(final Matrix m2);

    public abstract Matrix addColumns(final Matrix m2);

    public abstract Matrix addRows(final Matrix m2);

    public abstract List<Matrix> splitColumns(int numPieces);

    /* MUTABLE OPERATIONS */

    public DoubleMatrix2D data() {
        return m;
    }

    public double[][] toArray() {
        return m.toArray();
    }

    public Matrix set(final int i, final int j, final double value) {
        this.m.set(i, j, value);
        return this;
    }

    public DoubleMatrix1D row(final int row) {
        return this.m.viewRow(row);
    }

    public int dim() {
        return rows() * columns();
    }

    public int rows() {
        return m.rows();
    }

    public int columns() {
        return m.columns();
    }

    public double get(final int i, final int j) {
        return m.get(i, j);
    }

    public Matrix add(Matrix m2) {
        return apply(m2, ADD);
    }

    public Matrix subtract(Matrix m2) {
        return apply(m2, SUBTRACT);
    }

    public Matrix multiply(Matrix m2) {
        return apply(m2, MULTIPLY);
    }

    public Matrix multiply(double s) {
        return apply(new math.functions.Multiply(s));
    }

    public Matrix divide(Matrix m2) {
        return apply(m2, DIVIDE);
    }

    public Matrix divide(double s) {
        return apply(new math.functions.Divide(s));
    }

    public Matrix pow(double power) {
        return apply(new Power(power));
    }

    public double sum() {
        double sum = 0.0;
        for(int i = 0; i < rows(); i++) {
            for(int j = 0; j < columns(); j++) {
                sum += get(i, j);
            }
        }
        return sum;
    }

    public abstract Matrix apply(final DoubleFunction function);

    public abstract Matrix apply(final Matrix m2, final DoubleDoubleFunction function);

    public static List<double[][]> splitColumns(final Matrix m, int numPieces) {
        final List<double[][]> pieces = new ArrayList<>(numPieces);
        final int rows = m.rows();
        final int cols = m.columns() / numPieces; // must be evenly splittable
        for(int p = 0; p < numPieces; p++) {
            final double[][] piece = new double[rows][cols];
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    piece[i][j] = m.get(i, j + (p * cols));
                }
            }
            pieces.add(piece);
        }
        return pieces;
    }

    public static double[][] concatColumns(final Matrix... m) {
        int totalCols = 0;
        for(int i = 0; i < m.length; i++) {
            totalCols += m[i].columns();
        }
        final int rows = m[0].rows();
        final double[][] appended = new double[rows][totalCols];
        for(int k = 0; k < m.length; k++) {
            final int cols = m[k].columns();
            for(int i = 0; i < rows; i++) {
                final int start = k * cols;
                System.arraycopy(m[k].row(i).toArray(), 0, appended[i], start, cols);
            }
        }
        return appended;
    }


    public static int rows(final double[][] m) {
        return m.length;
    }

    public static int cols(final double[][] m) {
        return m[0].length;
    }

}
