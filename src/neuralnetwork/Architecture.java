package neuralnetwork;

import neuralnetwork.ActivationFunctions.Sigmoid;
import neuralnetwork.ActivationFunctions.ActivationFunction;
import java.util.Arrays;

/**
 * Describes the number of nodes in each layer and the number of weights.
 * Provides transformations from an index in a vector describing a neural
 * network to a weight or bias index defined by layer number, row, and height,
 * and vice versa. each node has.
 *
 * @author Dov Neimand
 */
public class Architecture {

    private final LayerArchitecture[] dims;
    private final ActivationFunction actFunc;

    /**
     * The constructor.
     *
     * @param af The activation function for this architecture.
     * @param dataDim The dimension / size of each data vector that will be fed
     * into the neural network.
     * @param numNodesPerLayer The number of nodes in each layer. Note, an
     * illegal argument exception is thrown if the number of nodes in the last
     * layer does not equal outputDim.
     */
    public Architecture(ActivationFunction af, int dataDim, int... numNodesPerLayer) {
        this.actFunc = af;

        if (numNodesPerLayer.length < 1)
            throw new IllegalArgumentException("The network must have at least"
                    + "one layer.");

        dims = new LayerArchitecture[numNodesPerLayer.length];

        dims[0] = new LayerArchitecture(numNodesPerLayer[0], dataDim, 0, this);
        for (int i = 1, startInd = dims[0].length(); i < numNodesPerLayer.length; i++, startInd += dims[i
                - 1].length())
            dims[i] = new LayerArchitecture(numNodesPerLayer[i], dims[i - 1].rows, startInd, this);

        if (Arrays.stream(numNodesPerLayer)
                .anyMatch(i -> i > dataDim && i > outputDim()
                || i < dataDim && i < outputDim()))
            System.err.println("neuralnetwork.NNLayerDims.<init>()"
                    + "\nIt is recomended each layer has a number of nodes "
                    + "between the dataDim and outputDim.");
    }

    /**
     * The activation function of the network.
     *
     * @return The activation function of the network.
     */
    public ActivationFunction getActFunc() {
        return actFunc;
    }

    /**
     * The number of nodes in the final layer.
     *
     * @return The number of nodes in the final layer.
     */
    public int outputDim() {
        return dims[dims.length - 1].rows;
    }

    /**
     * The dimensionality of the input data for the neural network.
     *
     * @return
     */
    public int inputDim() {
        return dims[0].cols;
    }

    /**
     * The dimensions of the requested layer.
     *
     * @param i The layer number for the desired dimensions.
     * @return The dimensions of the requested layer.
     */
    public LayerArchitecture get(int i) {
        return dims[i];
    }

    /**
     * The number of layers.
     *
     * @return
     */
    public int numLayers() {
        return dims.length;
    }

    /**
     * Finds the index in the vector that describes a neural networks weights
     * and biases.
     *
     * @param ind An index in layer, row, column notation
     * @return The index in the Rn vector describing a neural network with these
     * dimensions.
     */
    public int getIndex(Indices ind) {
        return getWeightIndex(ind.layer, ind.row, ind.col);
    }

    private int layerIndex(int layerIndex) {
        int index = 0;
        for (int i = 0; i < layerIndex; i++)
            index += dims[i].length();
        return index;
    }

    /**
     * Finds the index in the vector that describes a neural networks weights
     * and biases.
     *
     * @param matrix The index of the matrix in dims.
     * @param row The index of the desired row.
     * @param col The index of the desired column. This should be -1 for a bias.
     * @return The index in the Rn vector describing a neural network with these
     * dimensions.
     */
    public int getWeightIndex(int matrix, int row, int col) {

        return layerIndex(matrix) + (col != -1
                ? dims[matrix].rows * col
                : dims[matrix].numWeights()) + row;
    }

    /**
     * Gets the neural network vector's index for the proffered bias.
     *
     * @param layer The layer of the desired bias.
     * @param row the row of the desired bias.
     * @return The index in the vector that describes the nueral network.
     */
    public int getBiasIndex(int layer, int row) {
        return layerIndex(layer) + dims[layer].numWeights() + row;
    }

    /**
     * Gets the indices of a neural-network description-vector value.
     *
     * @param ind The index in the neural-network description vector
     * @return The indices of the value held in the neural-network description
     * vector.
     */
    public Indices getIndex(int ind) {
        int matrixInd = 0;
        int matrixStart = 0;
        while (matrixStart + dims[matrixInd].length() <= ind)
            matrixStart += dims[matrixInd++].length();

        int localInd = ind - matrixStart;

        if (ind >= matrixStart + dims[matrixInd].numWeights())
            return new Indices(matrixInd, ind - matrixStart
                    - dims[matrixInd].numWeights(), ind);
        else
            return new Indices(matrixInd, localInd % dims[matrixInd].rows, localInd
                    / dims[matrixInd].rows);
    }

    /**
     * Checks if the proffered index actually exists.
     *
     * @param ind An index whose validity needs to be checked.
     * @return True if the index might be valid, false if it is definitely not.
     */
    public boolean isIndex(Indices ind) {
        return (dims[ind.layer].rows > ind.row
                && dims[ind.layer].cols > ind.col)
                || (ind.isBias() && ind.row < dims[ind.layer].rows);
    }

    /**
     * The total number of variables, or alternatively, the total number of
     * weight and biases, the length of the vector that uniquely describes a
     * neural network with these layer dims.
     *
     * @return
     */
    public int numVariables() {
        return dims[dims.length - 1].startIndex + dims[dims.length - 1].length();
    }

    @Override
    public String toString() {
        return Arrays.toString(dims);
    }

    /**
     * Finds the layer with the most nodes and returns the number of nodes in 
     * that layer. This method has linear time as a function of the number of 
     * layers.
     * @return The number of nodes in the layer with the most nodes.
     */
    public int mostNodesInALayer() {
        int argMax = 0;
        for(int i = 1; i < dims.length; i++)
            if(dims[i].rows > dims[argMax].rows) argMax = i;
        return dims[argMax].rows;
    }

    /**
     * Runs some basic tests on this class.
     * @param args Not used.
     */
    public static void main(String[] args) {
        Architecture dims = new Architecture(new Sigmoid(), 2, 3, 2);
        System.out.println(dims.toString());
        for (int i = 0; i < dims.numVariables(); i++) {
//            if(!dims.isIndex(dims.getIndex(i))) throw new RuntimeException("Bad index generated at " + i);
            System.out.println(i + ": " + dims.getIndex(i));
        }

        System.out.println(dims.getWeightIndex(0, 0, 0));
        System.out.println(dims.getWeightIndex(0, 1, 0));
        System.out.println(dims.getWeightIndex(0, 2, 0));
        System.out.println(dims.getWeightIndex(0, 0, 1));
        System.out.println(dims.getWeightIndex(0, 1, 1));
        System.out.println(dims.getWeightIndex(0, 2, 1));
        System.out.println(dims.getBiasIndex(0, 0));
        System.out.println(dims.getBiasIndex(0, 1));
        System.out.println(dims.getBiasIndex(0, 2));
        System.out.println(dims.getWeightIndex(1, 0, 0));
        System.out.println(dims.getWeightIndex(1, 1, 0));
        System.out.println(dims.getWeightIndex(1, 0, 1));
        System.out.println(dims.getWeightIndex(1, 1, 1));
        System.out.println(dims.getWeightIndex(1, 0, 2));
        System.out.println(dims.getWeightIndex(1, 1, 2));
        System.out.println(dims.getBiasIndex(1, 0));
        System.out.println(dims.getBiasIndex(1, 1));

    }
}
