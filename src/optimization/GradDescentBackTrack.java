package optimization;

import java.util.concurrent.RecursiveTask;
import org.jblas.DoubleMatrix;

/**
 * Runs gradient descent back tracking.
 * @author Dov Neimand
 */
public class GradDescentBackTrack extends RecursiveTask<double[]>{

    /**
     * The rate at which the step size decreases.
     */
    protected final double gamma, 
            /**
             * Used to set the jump size.  See literature.
             */
            c,             
            /**
             * The tolerance for a correct answer.
             */
            tolerance;
    /**
     * The function being optimized.
     */
    protected DiffReal f;
    /**
     * A fist guess for what the minimum might be.
     */
    protected final DoubleMatrix start;

        
    /**
     * 
     * @param f The function to be minimized.  If this is stochastic, then the
     * stochastic method should be implemented.
     * @param tolerance The smaller this is, the more accurate the result will
     * 
     */
    public GradDescentBackTrack(DiffReal f, double tolerance) {
        this.gamma = .5;
        this.c = .5;
        this.tolerance = tolerance;
        this.f = f;
        start = DoubleMatrix.randn(f.domainDim());
    }

    /**
     * The backtracking portion of backtracking gradient descent.  The method
     * picks a point at some distance, and then reduces that distance by a 
     * factor of gamma until the value of f at that point is less than c * the
     * slope at the given point.
     * @param from The point we're jumping from.
     * @param atX The gradient of the from point.
     * @return a new point along the direction of grad that has improved
     * f more than the slope of grad times c.
     */
    protected DoubleMatrix jump(DoubleMatrix from, FuncAt atX) {
        double t = gamma;
        final double reducedSlope = c * atX.grad.dot(atX.grad);

        while (f.at(from.sub(atX.grad.mul(t))) > atX.val - t*reducedSlope)
            t *= gamma;
        
        DoubleMatrix to = from.add(atX.grad.mul(-t));
        
        if(from.sub(to).norm2() <= 1e-14) //TODO:remove
            throw new RuntimeException("This jump did not move at all.  The gradient is: " + atX);
        
        return to; 
    }
    
    
    /**
     * Finds the minimal value of the unconstrained function.
     * @return 
     */
    @Override
    public double[] compute() {
        
        DoubleMatrix x = start;
        FuncAt atX = f.funcAt(x.data);

        while (!atMin(atX.grad)) {
            
            x = jump(x, atX);
            f = f.stochastic();
            atX = f.funcAt(x.data);
        }

        return x.data;
    }
    
    /**
     * Is this the minimum point?
     * @param grad The gradient of the point.
     * @return True if this is a local minimum, false otherwise.
     */
    protected boolean atMin(DoubleMatrix grad){
        return grad.dot(grad) <= tolerance;
    }
    
    /**
     * Tests the optimization method on a simple function.
     * @param args No arguments are passed.
     */
    public static void main(String[] args) {
        DiffReal f = new DiffReal() {
            @Override
            public DoubleMatrix grad(double[] x) {
                return new DoubleMatrix(
                        new double[]{Math.cos(x[0])}
                );
            }

            @Override
            public int domainDim() {
                return 1;
            }

            @Override
            public double applyAsDouble(double[] x) {
                return Math.sin(x[0]);
            }
        };
        
        GradDescentBackTrack gdbt = new GradDescentBackTrack(f, 1e-10);
        
        System.out.println(-2*gdbt.compute()[0]);
    }

}
