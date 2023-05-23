package singleNode;

/**
 *
 * @author Dov Neimand
 */
public class GradDescentBackTrack {

    private final double gamma, c, tolerance;
    private final DiffReal f;

    /**
     * The constructor.
     * @param f The function to be minimized.
     */
    public GradDescentBackTrack(DiffReal f, double tolerance) {
        this.gamma = .5;
        this.c = .5;
        this.tolerance = tolerance;
        this.f = f;
    }

    /**
     * The backtracking portion of backtracking gradient descent.  The method
     * picks a point at some distance, and then reduces that distance by a 
     * factor of gamma until the value of f at that point is less than c * the
     * slope at the given point.
     * @param from The point we're jumping from.
     * @param grad The gradient of the from point.
     * @return a new point along the direction of grad that has improved
     * f more than the slope of grad times c.
     */
    private RVec jump(final RVec from, final RVec grad) {
        double t = gamma;
        final double fAtX = f.at(from), reducedSlope = c * grad.normSq();

        while (f.at(from.minus(grad.mult(t))) > fAtX - t*reducedSlope)
            t *= gamma;
        
        return from.plus(grad.mult(-t));
    }

    /**
     * Finds the minimal value of the unconstrained function.
     * @return 
     */
    public RVec run() {
        
        RVec x = new RVec(new double[f.domainDim()]);
        RVec grad = f.grad(x);

        while (grad.normSq() > tolerance) {
            
            x = jump(x, grad);
            grad = f.grad(x);
        }

        return x;
    }
    
    /**
     * Tests the optimization method on a simple function.
     * @param args No arguments are passed.
     */
    public static void main(String[] args) {
        DiffReal f = new DiffReal() {
            @Override
            public RVec grad(RVec x) {
                return new RVec(Math.cos(x.at(0)));
            }

            @Override
            public int domainDim() {
                return 1;
            }

            @Override
            public double applyAsDouble(RVec x) {
                return Math.sin(x.at(0));
            }
        };
        
        GradDescentBackTrack gdbt = new GradDescentBackTrack(f, .1);
        
        System.out.println(2*gdbt.run().at(0));
    }

}
