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
    public DoubleMatrix grad(double[] x);
    
    
    
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
     * The gradient and value of the function at x.
     * @param x The point the gradient and value are taken from.
     * @return The gradient and value of the function at x.
     */
    public default FuncAt funcAt(double[] x){
        return new FuncAt(grad(x), at(x));
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
