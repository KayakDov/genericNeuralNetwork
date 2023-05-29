package optimization;

import org.jblas.DoubleMatrix;

/**
 * This class holds information about a function at a specific point.
 * @author Dov Neimand
 */
public class FuncAt {

    /**
     * The gradient of the function at the point.
     */
    public final DoubleMatrix grad;
    
    /**
     * The value of the function at the point.
     */
    public double val;

    public FuncAt(DoubleMatrix grad, double val) {
        this.grad = grad;
        this.val = val;
    }
    
    /**
     * Adds the proffered values to this FuncAt in place.
     * @param fAtX The values to be added to these values.
     * @return This instance.
     */
    public FuncAt addi(FuncAt fAtX){
        grad.addi(fAtX.grad);
        val += fAtX.val;
        return this;
    }
    
}
