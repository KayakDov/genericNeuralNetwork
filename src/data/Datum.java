
package data;

import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class Datum extends DoubleMatrix{
    public final int type, numTypes;

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
