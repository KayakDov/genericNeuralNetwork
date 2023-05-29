package neuralnetwork;

/**
 * The indices of a given weight or bias.
 * @author Dov Neimand
 */
public class Indices {
    
    public final int layer, row, col, source;

    /**
     * A constructor for a weight or bias.
     * @param matrix The index of the layer of the weight.
     * @param row The index of the row of the weight.
     * @param col The index of the column of the weight.  This can be set to 
     * -1 if a bias is meant to be constructed.
     */
    public Indices(int matrix, int row, int col, int source) {
        this.layer = matrix;
        this.row = row;
        this.col = col;
        this.source = source;
    }
    
    /**
     * A constructor for a bias.
     * @param matrix The index of the layer of the weight.
     * @param row The index of the row of the weight.
     */
    public Indices(int matrix, int row, int source) {
        this.layer = matrix;
        this.row = row;
        this.col = -1;
        this.source = source;
    }
    
    /**
     * Is this a weight?
     * @return True if this is a weight, false if this is a bias.
     */
    public boolean isWeight(){
        return col != -1;
    }
    
    /**
     * Is this a bias?
     * @return True if col == -1, false otherwise.
     */
    public boolean isBias(){
        return col == -1;
    }

    @Override
    public String toString() {
        return "("+ layer + ", "  +  row + (isWeight()? ", "+col:"") + ")";
    }
    
    
}
