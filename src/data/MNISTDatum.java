
package data;

/**
 *
 * @author Dov Neimand
 */
public class MNISTDatum extends Datum{

    public final static int NUM_DIGITS = 10;
    
    public MNISTDatum(double[] x, int type) {
        super(x, type, NUM_DIGITS);
    }

}
