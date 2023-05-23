package singleNode;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleToIntFunction;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;

/**
 *
 * @author Dov Neimand
 */
public class RVec {

    /**
     * A random number generator to be broadly available.
     */
    public static Random rand = new Random();

    protected final double[] x;

    public RVec(double... x) {
        this.x = x;
    }

    /**
     * The value of the vector at the given index.
     *
     * @param i the index of the desired scalar.
     * @return the real value of the vector at the given index.
     */
    public double at(int i) {
        return x[i];
    }

    /**
     * The dimension of the vector.
     *
     * @return The dimension of the vector.
     */
    public int dim() {
        return x.length;
    }

    /**
     * The dot product of two vectors with the same length.
     *
     * @param v The other Vector.
     * @return The dot product of the two vectors.
     */
    public double dot(RVec v) {
        if (dim() != v.dim())
            throw new RuntimeException("The length of the two vectors is "
                    + "not equal. " + "dim" + toString() + ") == "+ dim() + 
                    " != " + v.dim() + " = dim" + v.toString());
        
        return IntStream.range(0, dim()).mapToDouble(i -> at(i) * v.at(i)).sum();
    }

    /**
     * Builds a real vector with the proffered function.
     *
     * @param dim The dimension of the new vector.
     * @param f Constructs the value of each element from the index of the
     * element.
     */
    public RVec(int dim, IntToDoubleFunction f) {
        x = new double[dim];
        Arrays.setAll(x, f);
    }

    /**
     * The sum of this vector and the proffered vector.
     *
     * @param v The vector to be added to this one.
     * @return A new vector that is the sum of this vector and the proffered
     * vector.
     */
    public RVec plus(RVec v) {
        return new RVec(dim(), i -> at(i) + v.at(i));
    }
    
    /**
     * Adds the proffered value to this array at the specified index.
     * @param i The index the proffered value is to be added to.
     * @param val The value to be added at the given index.
     * @return A new vector equal to this one except at index i where val is
     * added to the value at the given index.
     */
    public RVec plus(int i, double val){
        RVec plus = new RVec(dim(), j -> j);
        plus.x[i] += val;
        return plus;
    }

    /**
     * The product of this vector and the proffered scalar.
     *
     * @param r A value to be multiplied by every value in this vector.
     * @return A new vector equal to this vector times r.
     */
    public RVec mult(double r) {
        return new RVec(dim(), i -> at(i) * r);
    }

    /**
     * The difference between this vector and the proffered vector.
     *
     * @param v the vector to be subtracted from this vector.
     * @return A new vector that is the difference between the two vectors.
     */
    public RVec minus(RVec v) {
        return plus(v.mult(-1));
    }

    /**
     * The distance between this vector and the proffered vector, squared.
     *
     * @param v A vector as some distance from this vector.
     * @return The distance to v squared.
     */
    public double distSquared(RVec v) {
        RVec dif = minus(v);
        return dif.dot(dif);
    }

    /**
     * The norm of this vector squared.
     *
     * @return The norm of this vector squared.
     */
    public double normSq() {
        return dot(this);
    }

    /**
     * The norm of this vector.
     *
     * @return The norm of this vector.
     */
    public double norm() {
        return Math.sqrt(normSq());
    }

    /**
     * Returns a random vevtor in the unit sphere.
     *
     * @param dim The dimension of the vector.
     * @return A random unit vector in the unit sphere.
     */
    public static RVec random(int dim) {
        RVec inCube = new RVec(dim, i -> rand.nextGaussian());
        return inCube.mult(1/inCube.norm()).mult(rand.nextDouble());
    }
    
    /**
     * A subsequence of this vector.
     * @param start The start index of the new vector, inclusive.
     * @param end The end index of the subsequence, exclusive.
     */
    public RVec subVector(int start, int end){
        return new RVec(end - start, i -> at(i + start));
    }

    @Override
    public String toString() {
        return Arrays.toString(x).replace('[', '(').replace(']', ')');
    }
    
    
}
