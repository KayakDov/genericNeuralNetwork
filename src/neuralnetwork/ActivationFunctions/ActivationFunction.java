package neuralnetwork.ActivationFunctions;

import java.io.Serializable;
import java.util.function.DoubleUnaryOperator;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public interface ActivationFunction extends DoubleUnaryOperator, Serializable{

    /**
     * The derivative of this function.
     *
     * @param x The value at which the derivative is to be calculated.
     * @return The derivative of this function at x.
     */
    public double ddt(double x);

    /**
     * The derivative of this method applied row wise to a vector.
     *
     * @param vector The vector for which the derivative is desired.
     * @return A new vector that is the row wise derivative of the proffered
     * vector.
     */
    public default DoubleMatrix ddt(DoubleMatrix vector) {
        return map(vector, v -> ddt(v));
    }

    /**
     * A mapping of the vector. //TODO: an in place version of this needs to be
     * made.
     *
     * @param vector The vector to be mapped. The vector remains unchanged.
     * @param f The mapping.
     * @return The result of the mapping.
     */
    private static DoubleMatrix map(DoubleMatrix vector, DoubleUnaryOperator f) {
        return map(vector, new DoubleMatrix(vector.length), f);
    }

    /**
     * A mapping of the vector in place.
     *
     * @param from The vector to be mapped. The vector is changed.
     * @param to The vector the mapping is put into.
     * @param f The mapping.
     * @return The result of the mapping.
     */
    private static DoubleMatrix map(DoubleMatrix from, DoubleMatrix to, DoubleUnaryOperator f) {
        for (int i = 0; i < from.length; i++)
            to.data[i] = f.applyAsDouble(from.data[i]);
        return to;
    }

    /**
     * The value of this function at val.
     *
     * @param d An value in the domain of the function.
     * @return The value of this function at val.
     */
    public default double apply(double d) {
        return applyAsDouble(d);
    }

    /**
     * This method applied row wise to a vector.
     *
     * @param vector The vector this method is applied row wise to.
     * @return A new vector where each row is this function applied to the
     * elements of vector.
     */
    public default DoubleMatrix apply(DoubleMatrix vector) {
        return map(vector, this);
    }
    
    /**
     * This method applied row wise in place to a vector.
     *
     * @param vector The vector this method is applied row wise to.
     * @return A new vector where each row is this function applied to the
     * elements of vector.
     */
    public default DoubleMatrix applyi(DoubleMatrix vector) {
        return map(vector, vector, this);
    }

    /**
     * This class encapsulates the value of a function and its derivative at
     * some operand. Typing is not used to avoid boxing.
     */
    public class AtScalar {

        /**
         * The value of this function at the operand.
         */
        public final double val;
        /**
         * The value of the derivative of this function at the operand.
         */
        public final double ddt;

        public AtScalar(double d, double ddt) {
            this.val = d;
            this.ddt = ddt;
        }

    }

    /**
     * This class encapsulates the value of a function and its derivative at
     * some operand. Typing is not used to avoid boxing.
     */
    public class AtVector {

        /**
         * The value of this function at the operand.
         */
        public final DoubleMatrix val;
        /**
         * The value of the derivative of this function at the operand.
         */
        public final DoubleMatrix ddt;

        public AtVector(DoubleMatrix d, DoubleMatrix ddt) {
            this.val = d;
            this.ddt = ddt;
        }

    }

    /**
     * The derivative and value of this function at x. This method is provided
     * to allow an implementation that avoids redundant calculations between the
     * derivative and the value.
     *
     * @param x The operand.
     * @return The derivative and value of this function at x.
     */
    public default AtScalar at(double x) {
        return new AtScalar(apply(x), ddt(x));
    }

    /**
     * The derivative and value of this function at x. This method is provided
     * to allow an implementation that avoids redundant calculations between the
     * derivative and the value.
     *
     * @param x The operand.
     * @return The derivative and value of this function at x.
     */
    public default AtVector at(DoubleMatrix x) {
        return new AtVector(apply(x), ddt(x));
    }

    /**
     * The derivative and value of this function at x, with the value computed
     * in place. This method is provided to allow an implementation that avoids
     * redundant calculations between the derivative and the value.
     *
     * @param x The operand.
     * @return The derivative and value of this function at x.
     */
    public default AtVector ati(DoubleMatrix x) {
        return new AtVector(apply(x), ddt(x));
    }
}
