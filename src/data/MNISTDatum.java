
package data;

/**
 *
 * @author Dov Neimand
 */
public class MNISTDatum extends Datum{

    /**
     * The total number of classifications / digits.
     */
    public final static int NUM_DIGITS = 10;
    
    /**
     * The constructor.
     * @param x The data values.
     * @param type The classification of this datum.
     */
    public MNISTDatum(double[] x, int type) {
        super(x, type, NUM_DIGITS);
    }

}
