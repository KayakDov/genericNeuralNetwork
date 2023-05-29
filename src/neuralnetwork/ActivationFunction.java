
package neuralnetwork;



import java.util.function.DoubleUnaryOperator;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public interface ActivationFunction extends DoubleUnaryOperator{
    
    /**
     * The derivative of this function.
     * @param x The value at which the derivative is to be calculated.
     * @return The derivative of this function at x.
     */
    public double ddt(double x);
    
    /**
     * The derivative of this method applied row wise to a vector.
     * @param vector The vector for which the derivative is desired.
     * @return A new vector that is the row wise derivative of the proffered 
     * vector.
     */
    public default DoubleMatrix ddt(DoubleMatrix vector){
        return map(vector, v -> ddt(v));
    }
        
    /**
     * A mapping of the vector. //TODO: an in place version of this needs
     * to be made.
     * @param vector The vector to be mapped.  The vector remains unchanged.
     * @param f The mapping.
     * @return The result of the mapping.
     */
    private static DoubleMatrix map(DoubleMatrix vector, DoubleUnaryOperator f){
        DoubleMatrix map = new DoubleMatrix(vector.length);
        for(int i = 0; i < vector.length; i++)
            map.data[i] = f.applyAsDouble(vector.data[i]);
        return map;
    }
    
    
    /**
     * The value of this function at d.
     * @param d An value in the domain of the function.
     * @return The value of this function at d.
     */
    public default double apply(double d){
        return applyAsDouble(d);
    }
    
    /**
     *  This method applied row wise to a vector.
     * @param vector The vector this method is applied row wise to.
     * @return A new vector where each row is this function applied to the 
     * elements of vector.
     */
    public default DoubleMatrix apply(DoubleMatrix vector){
        return map(vector, this);
    }
}
