package math;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kenny on 5/24/14.
 */
public class SparseMatrix extends Matrix {

    protected SparseMatrix(DoubleMatrix2D doubleMatrix2D) {
        super(doubleMatrix2D);
    }

    /* IMMUTABLE OPERATIONS */

    @Override
    public Matrix copy() {
        return make(m.copy());
    }

    @Override
    public Matrix dot(Matrix m2) {
        return make(DENSE_DOUBLE_ALGEBRA.mult(m, m2.data()));
    }

    @Override
    public Matrix transpose() {
        return make(DENSE_DOUBLE_ALGEBRA.transpose(this.m));
    }

    @Override
    public Matrix addColumns(final Matrix m2) {
        return make(DoubleFactory2D.sparse.appendColumns(m, m2.data()));
    }

    @Override
    public Matrix addRows(final Matrix m2) {
        return make(DoubleFactory2D.sparse.appendRows(m, m2.data()));
    }

    @Override
    public List<Matrix> splitColumns(int numPieces) {
        List<double[][]> pieces = Matrix.splitColumns(this, numPieces);
        List<Matrix> mPieces = new ArrayList<>(pieces.size());
        for(double[][] piece : pieces) {
            mPieces.add(DenseMatrix.make(piece));
        }
        return mPieces;
    }

    /* MUTABLE OPERATIONS */
    @Override
    public Matrix apply(DoubleFunction function) {
        return make(m.assign(function));
    }

    @Override
    public Matrix apply(Matrix m2, DoubleDoubleFunction function) {
        return make(m.assign(m2.data(), function));
    }

    public static Matrix make(DoubleMatrix2D m) {
        return new SparseMatrix(m);
    }

    public static Matrix make(int r, int c) {
        return new SparseMatrix(DoubleFactory2D.sparse.make(r, c));
    }

    public static Matrix randomGaussian(int r, int c) {
        return new SparseMatrix(DoubleFactory2D.sparse.make(r, c).assign(RANDOM_GAUSSIAN));
    }

    public static Matrix random(int r, int c) {
        return new SparseMatrix(DoubleFactory2D.sparse.make(r, c).assign(RANDOM_DOUBLE));
    }

    public static Matrix make(double[][] m) {
        return new SparseMatrix(DoubleFactory2D.sparse.make(m));
    }

    @Override
    public String toString() {
        return m.toString();
    }

}
