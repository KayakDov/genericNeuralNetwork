package optimization;

import java.util.function.ToDoubleFunction;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Kayak
 */
public interface DiffReal extends ToDoubleFunction<double[]>{
    
    
    /**
     * The gradient at this function at x.
     * @param x The point at which the gradient is calculated.
     * @return The gradient at the function at x.
     */
    public abstract DoubleMatrix grad(double[] x);
    
    
    /**
     * The gradient at this function at x.
     * @param x The point at which the gradient is calculated.
     * @return The gradient at the function at x.
     */
    public default DoubleMatrix grad(DoubleMatrix x){
        return grad(x.data);
    }
    
    /**
     * This is the same as the applyAsDouble function.
     * @param x A point in the domain at the function.
     * @return The value at the function at x;
     */
    public default double at(double... x){
        return applyAsDouble(x);
    }
    
    /**
     * This is the same as the applyAsDouble function.
     * @param x A point in the domain at the function.
     * @return The value at the function at x;
     */
    public default double at(DoubleMatrix x){
        return at(x.data);
    }
    
    public abstract int domainDim();
}
