package myneuralnetwork;

/**
 * This class holds a datum and a binary classification for that datam.
 * @author Dov Neimand
 */
public class Datum extends RVec{
    private boolean isClassified;
    private int type;

    /**
     * The constructor.
     * This datum will not be labeled.
     * @param values The values of this datum.
     */
    public Datum(double[] values) {
        super(values);
        isClassified = false;
    }
    
    /**
     * The Constructor.
     * @param values The values of the data point.
     * @param classification The classification of this datum.
     */
    public Datum(double[] values, int classification){
       this(values);
       isClassified = true;
        this.type = classification;
    }
    
    /**
     * The Constructor.
     * @param values The values of the data point.
     * @param classification The classification of this datum.
     */
    public Datum(RVec values, int classification){
       this(values.x, classification);
    }
    
    /**
     * The Constructor.
     * @param values The values of the data point.
     */
    public Datum(RVec values){
       this(values.x);
    }
    
    /**
     * The Constructor.
     * @param values The values of the data point.
     * @param classification The classification of this datum.
     */
    public Datum(RVec values, boolean classification){
       this(values.x, classification?1:0);
    }

    /**
     * Has this datum been classified.
     * @return  True if the datum has been classified, and false otherwise.
     */
    public boolean isClassified() {
        return isClassified;
    }

    /**
     * The classification of this datum.
     * @return The classification of this datum.
     */
    public int getType() {
        return type;
    }
    
    /**
     * Classify this data.  Note that it is forbidden to reclassify a datum.
     * @param classification The classification being given to this datum.
     */
    public void classify(int classification){
        if(isClassified) throw new RuntimeException("You are attempting to classify data that is already classified.");
        isClassified = true;
        this.type = classification;
    }
    
}
