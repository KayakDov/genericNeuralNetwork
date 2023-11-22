
package data;

import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class Datum extends DoubleMatrix{
    /**
     * The classification of this datum.
     */
    public final int type, 
            /**
             * The total number of classifications that exist?
             */            
            numTypes;

    /**
     * The constructor.
     * @param x The values in this datum.
     * @param type the classification of this datum.
     * @param numTypes The total number of types this datum can have.
     */
    public Datum(double[] x, int type, int numTypes) {
        super(x);
        this.type = type;
        this.numTypes = numTypes;
    }
    
    /**
     * The optimal output a neural network would produce for this datum.
     * @return The optimal output a neural network would produce for this datum.
     */
    public final DoubleMatrix type(){
        double[] type = new double[numTypes];
        type[this.type] = 1;
        return new DoubleMatrix(type);
    }

}
