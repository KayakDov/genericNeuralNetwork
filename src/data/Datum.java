
package Data;

import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class Datum extends DoubleMatrix{
    public final int type;

    public Datum(double[] x, int type) {
        super(x);
        this.type = type;
    }
    
    /**
     * The optimal output a neural network would produce for this datum.
     * @return The optimal output a neural network would produce for this datum.
     */
    public final DoubleMatrix type(){
        double[] type = new double[data.length];
        type[this.type] = 1;
        return new DoubleMatrix(type);
    }

}
