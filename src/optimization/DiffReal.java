package optimization;

import java.util.function.ToDoubleFunction;
import singleNode.RVec;

/**
 *
 * @author Kayak
 */
public interface DiffReal extends ToDoubleFunction<RVec>{
    
    
    /**
     * The gradient at this function at x.
     * @param x The point at which the gradient is calculated.
     * @return The gradient at the function at x.
     */
    public abstract RVec grad(RVec x);
    
    
    /**
     * The gradient at this function at x.
     * @param x The point at which the gradient is calculated.
     * @return The gradient at the function at x.
     */
    public default RVec grad(double ... x){
        return grad(new RVec(x));
    }
    
    /**
     * This is the same as the applyAsDouble function.
     * @param x A point in the domain at the function.
     * @return The value at the function at x;
     */
    public default double at(RVec x){
        return applyAsDouble(x);
    }
    
    /**
     * This is the same as the applyAsDouble function.
     * @param x A point in the domain at the function.
     * @return The value at the function at x;
     */
    public default double at(double ... x){
        return at(new RVec(x));
    }
    
    public abstract int domainDim();
}
