package neuralnetwork;

import neuralnetwork.ActivationFunctions.ActivationFunction;
import data.Datum;
import java.util.function.Function;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class Layer implements Function<DoubleMatrix, DoubleMatrix> {

    private DoubleMatrix weights, bias;
    public final ActivationFunction actFunc;
    public final Layer subLayer;
    public final LayerArchitecture arch;

    /**
     * Does this layer have a sublayer?
     *
     * @return True if the layer has a sub layer, false otherwise.
     */
    public boolean hasSubLayer() {
        return subLayer != null;
    }

    /**
     * Construct a trained layer of neural nodes.
     *
     * @param weights Each row is the weight of a node.
     * @param bias Each row is the bias of a node.
     * @param subLayer The layer this layer is called on. This should be null if
     * this layer is called directly on the data.
     * @layerArch The architecture for this layer.
     */
    private Layer(DoubleMatrix weights, DoubleMatrix bias, ActivationFunction af, Layer subLayer, LayerArchitecture layerArch) {
        this.weights = weights;
        this.bias = bias;
        this.actFunc = af;
        this.subLayer = subLayer;
        arch = layerArch;
    }

    /**
     * Applies a node in the sublayer.
     *
     * @param row The node to be applied.
     * @param x The datum on which the node is applied.
     * @return The value of the sublayer node on the datum.
     */
    public double subLayer(DoubleMatrix x, int row) {
        if (hasSubLayer()) return subLayer.applyNode(x, row);
        return x.get(row);
    }
    
    /**
     * A container for the results of backtracking.  
     */
    public class BackTrackResult{
        public final DoubleMatrix grad;
        public final DoubleMatrix apply;

        public BackTrackResult(DoubleMatrix grad, DoubleMatrix result) {
            this.grad = grad;
            this.apply = result;
        }
    }
    
    /**
     * The gradient of this layer as a function of the weights and biases 
     * applied to x.
     * @param x The point the neural network is being applied to.  For the 
     * purposes of computing the gradient, this is considered a constant.
     * @return The gradient of this method and the result of applying the layer
     * to x.  In the gradient vector, each row is the gradient over all the 
     * weights and biases for one of the highest layer nodes.
     */
    public BackTrackResult grad(Datum x){
        
        BackTrackResult btr = hasSubLayer()?
                subLayer.grad(x):
                new BackTrackResult(DoubleMatrix.EMPTY, x);
        
        DoubleMatrix grad = 
                new DoubleMatrix(arch.rows, arch.startIndex + arch.length());
        
        int w = 0;
        for(;w < btr.grad.columns; w++) 
            grad.putColumn(w, weights.mmul(btr.grad.getColumn(w)));
        
        for(int col = 0; col < weights.columns; col++) //indecies ordered (0,0), (1,0), ..., (n, 0), (1,0), (1,1), ..., (1,n), ...
            for(int row = 0; row < weights.rows; row++, w++)
                grad.putColumn(w, valAt(row, weights.rows, btr.apply.get(col)));
        
        for(int row = 0; row < weights.rows; row++, w++){
            grad.putColumn(w, valAt(row, weights.rows, 1));
        }

        ActivationFunction.AtVector actFuncAt = actFunc.ati(affineTransf(btr.apply));
        
        return new BackTrackResult(
                grad.mulColumnVector(actFuncAt.ddt), 
                actFuncAt.val
        );        
    }

    /**
     * Creates a vector of 0's except at the proffered index which holds val.
     *
     * @param val The lone non zero value in the vector.
     * @param index The index of the non zero value.
     * @param length The length of the vector.
     * @return A new vector of zeros with val at the given index.
     */
    private DoubleMatrix valAt(int index, int length, double val) {
        DoubleMatrix vec = new DoubleMatrix(length);
        vec.data[index] = val;
        return vec;
    }

    /**
     * Creates a neural layer from a vector
     *
     * @param vector The vector of all the weights and biases for the neural
     * network.
     * @param ld The arch for this layer.
     * @param sub The sublayer of this layer.
     */
    public Layer(double[] vector, LayerArchitecture ld, ActivationFunction af, Layer sub) {
        this(new DoubleMatrix(ld.rows, ld.cols), new DoubleMatrix(ld.rows), af, sub, ld);
        int numWeights = ld.numWeights();
        System.arraycopy(vector, ld.startIndex, weights.data, 0, numWeights);
        System.arraycopy(vector, ld.startIndex + numWeights, bias.data, 0, bias.length);

    }

    /**
     * The affine transformation defined by the wieghts matrix times vec plus
     * the bias. Wx+b.
     *
     * @param vec The vector to undergo transformation. The vector is not
     * changed.
     * @return A new vector that is the apply of the affine transformation
 applied to vec.
     */
    public DoubleMatrix affineTransf(DoubleMatrix vec) {
        return (weights.mmul(vec)).add(bias);
    }

    /**
     * A specific row of the affine transformation applied to the given vector.
     * The method is implemented directly for speed.
     *
     * @param vec The vector the affine transformation is applied to.
     * @param row The desired row of the apply.
     * @return The requested row of the affine transformation applied to vec.
     */
    public double affineTransf(DoubleMatrix vec, int row) {
        double affTrans = 0;

        for (int i = 0; i < vec.length; i++)
            affTrans += vec.get(i) * weights.get(row, i);

        return affTrans + bias.get(row);
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix vec) {
        return actFunc.apply(affineTransf(operand(vec)));
    }

    /**
     * The operand of this layer. If the layer has a sublayer, then this is
     * sublayer applied to x, otheriwse it is just x.
     *
     * @param x The datum the neural network is applied to.
     * @return The operand of this layer.
     */
    public DoubleMatrix operand(DoubleMatrix x) {
        if (hasSubLayer()) return subLayer.apply(x);
        return x;
    }
    
    /**
     * Applies a single node in this layer to the given datum.
     *
     * @param vec The datum that a node in this layer is t be applied to.
     * @param node The index of the node in this layer that is to be used.
     * @return The value of the indexed node in this layer over the datum.
     */
    public double applyNode(DoubleMatrix vec, int node) {
        return actFunc.apply(affineTransf(operand(vec), node));
    }

    /**
     * Sets the weights of the layer. Each row of the proffered matrix is the
     * weights of a single node. TODO:Use this instead of generating new layers.
     *
     * @param weights The weights of the nodes. Each row is the weights for a
     * single node.
     */
    public void setWeights(DoubleMatrix weights) {
        this.weights = weights;
    }

    /**
     * Sets the biases of the layer. TODO:Use this instead of generating new
     * layers.
     *
     * @param bias The new biases of the layer. Each value is the bias of a a
     * single node, whose weights are correspond to the nodes on the same rows
     * of the weights matrix.
     */
    public void setBias(DoubleMatrix bias) {
        this.bias = bias;
    }

    /**
     * The number of nodes in this layer.
     *
     * @return The number of nodes in this layer.
     */
    public int numNodes() {
        return bias.length;
    }

    /**
     * The weights of this layer.
     *
     * @return The weights of this layer.
     */
    public DoubleMatrix getWeights() {
        return weights;
    }

    /**
     * The biases of this layer.
     *
     * @return The biases of this layer.
     */
    public DoubleMatrix getBias() {
        return bias;
    }

    /**
     * The sum of the number of weights and biases in this layer.
     *
     * @return The sum of the number of weights and biases in this layer.
     */
    public int numberWeightsAndBiases() {
        return weights.length + bias.length;
    }

    @Override
    public String toString() {
        return "activation function " + actFunc.toString() + "\n"
                + weights.toString() + "x + " + bias.toString();
    }

}
