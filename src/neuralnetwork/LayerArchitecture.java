
package neuralnetwork;

import java.io.Serializable;

/**
 * This class describes the dimensions of a layer of a neural network.
 * @author Dov Neimand
 */
public class LayerArchitecture implements Serializable{
    /**
     * Each row is meant to represent a node, and the number of columns is the
     * number of weights in each node.  The number of columns does not include
     * the bias.
     */
    public final int rows, cols, startIndex;
    /**
     * The architecture for this layer.
     */
    public final Architecture networkArchitecture;

    /**
     * 
     * @param numNodes The number of nodes in the layer.  This is the number
     * or rows in the layer's matrix.
     * @param startIndex The index this networks weights start at in the 
     * networks weight vector
     * @param numWightsPerNode The number of weights each node has. This is the
     * number of columns int the layer's matrix.
     * @param network The architecture for this layer.
     */
    public LayerArchitecture(int numNodes, int numWightsPerNode, int startIndex, Architecture network) {
        this.rows = numNodes;
        this.cols = numWightsPerNode;
        this.startIndex = startIndex;
        networkArchitecture = network;
    }
    
    /**
     * The number of values held in the layer.
     * @return The number of values held in the layer.
     */
    public int length(){
        return numWeights() + rows;
    }
    
    /**
     * The total number of weights.
     * @return The total number of weights.
     */
    public int numWeights(){
        return rows*cols;
    }

    @Override
    public String toString() {
        return rows + " x " + cols;
    }
    
    
    
}
