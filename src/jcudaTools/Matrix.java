package jcudaTools;

import java.lang.ref.Cleaner;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import jcuda.*;
import jcuda.driver.CUdevice;
import jcuda.runtime.*;
import jcuda.jcublas.*;

/**
 * Represents a matrix stored on the GPU.
 */
public class Matrix {

    private final int height;
    private final int width;
    private final Pointer data; // Pointer to data on the GPU

    private static final ReferenceQueue<Matrix> referenceQueue = new ReferenceQueue<>();
    private static final Cleaner cleaner = createCleaner();

    // Static block for JCublas initialization
    static {
        JCublas.cublasInit();
        JCuda.setExceptionsEnabled(true);

    }

    private static double[][] transpose(double[][] matrix) {
        double[][] transp = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < transp.length; i++) {
            int k = i;
            Arrays.setAll(transp[i], j -> matrix[j][k]);
        }
        return transp;
    }

    public Matrix(double[][] matrix) {
        this(
                pointer(matrix[0].length * matrix.length * Sizeof.DOUBLE),
                matrix[0].length,
                matrix.length
        );

        for (int i = 0; i < width; i++) {

            Pointer devicePointer = data.withByteOffset(i * height * Sizeof.DOUBLE);

            JCuda.cudaMemcpy(
                    devicePointer,
                    Pointer.to(matrix[i]),
                    height * Sizeof.DOUBLE,
                    cudaMemcpyKind.cudaMemcpyHostToDevice
            );

        }

        cleaner.register(
                new PhantomReference<>(this, referenceQueue),
                () -> {
                    JCuda.cudaFree(data);
                    System.out.println("jcudaTools.Matrix.<init>(): memory freed.");
                }
        );
    }

    /**
     * Constructs a matrix from a Pointer to existing data on the GPU.
     *
     * @param data Pointer to data on the GPU.
     * @param height The height of the matrix.
     * @param width The width of the matrix.
     */
    private Matrix(Pointer data, int height, int width) {
        this.height = height;
        this.width = width;
        this.data = data;
    }

    /**
     * Gets the height of the matrix.
     *
     * @return The height of the matrix.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Gets the width of the matrix.
     *
     * @return The width of the matrix.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Gets the total number of elements in the matrix.
     *
     * @return The total number of elements in the matrix.
     */
    public int getSize() {
        return height * width;
    }

    /**
     * Performs matrix multiplication with another matrix using JCublas.
     *
     * @param other The other matrix to multiply with.
     * @return The result of matrix multiplication.
     */
    public Matrix multiply(Matrix other) {
        if (getWidth() != other.getHeight())
            throw new IllegalArgumentException("Matrix dimensions are not compatible for multiplication");

        Pointer result = pointer(getHeight() * other.getWidth());

        // Perform matrix multiplication
        JCublas.cublasDgemm(
                'n',
                'n',
                getHeight(),
                other.getWidth(),
                getWidth(),
                1.0,
                data,
                getHeight(),
                other.data,
                other.getHeight(),
                0.0,
                result,
                getHeight()
        );

        return new Matrix(result, getHeight(), other.getWidth());
    }

    /**
     * A pointer to a double vector with the given number of elements.
     *
     * @param numDoubles The number of elements in the vector.
     * @return A pointer to a vector of doubles.
     */
    private static Pointer pointer(int numDoubles) {
        Pointer p = new Pointer();
        JCuda.cudaMalloc(p, numDoubles * Sizeof.DOUBLE);
        return p;
    }

    /**
     * Performs element-wise addition with another matrix.
     *
     * @param other The other matrix to add.
     * @return The result of element-wise addition.
     */
    public Matrix add(Matrix other) {
        if (getHeight() != other.getHeight() || getWidth() != other.getWidth()) {
            throw new IllegalArgumentException("Matrix dimensions are not compatible for addition");
        }

        // Allocate memory for the result matrix on the GPU
        Pointer result = pointer(getSize());

        // Perform element-wise addition
        JCublas.cublasDaxpy(getSize(), 1.0, data, 1, result, 1); // Add current matrix to result
        JCublas.cublasDaxpy(getSize(), 1.0, other.data, 1, result, 1); // Add other matrix to result

        return new Matrix(result, getHeight(), getWidth());
    }

    /**
     * Returns a string representation of the matrix.
     *
     * @return The string representation of the matrix.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int row = 0; row < getHeight(); row++) {
            sb.append("[");
            for (int col = 0; col < getWidth(); col++) {
                sb.append(getElement(row, col));
                if (col < getWidth()- 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
            if (row < getHeight()- 1) {
                sb.append(",\n ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Gets the element at the specified row and column.
     *
     * @param row The row index.
     * @param column The column index.
     * @return The element at the specified row and column.
     */
    private double getElement(int row, int column) {
        double[] hostData = new double[1];
        JCuda.cudaMemcpy(
                Pointer.to(hostData),
                data.withByteOffset((row + column*getHeight()) * Sizeof.DOUBLE),
                Sizeof.DOUBLE,
                cudaMemcpyKind.cudaMemcpyDeviceToHost
        );
        return hostData[0];
    }

    /**
     * Creates a cleaner instance with a dedicated thread factory.
     *
     * @return The created cleaner instance.
     */
    private static Cleaner createCleaner() {
        ThreadFactory threadFactory = Executors.defaultThreadFactory();
        return Cleaner.create(threadFactory);
    }

    public static void main(String[] args) {

        Matrix matrixA = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix matrixB = new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});

        // Print the original matrices
        System.out.println("Matrix A:");
        System.out.println(matrixA);
        System.out.println();

        System.out.println("Matrix B:");
        System.out.println(matrixB);
        System.out.println();
//
//        // Test matrix multiplication
        Matrix product = matrixA.multiply(matrixB);
        System.out.println("Matrix Multiplication Result:");
        System.out.println(product);
        System.out.println();
        // Test matrix addition
//        Matrix sum = matrixA.add(matrixB);
//        System.out.println("Matrix Addition Result:");
//        System.out.println(sum);
    }

}
