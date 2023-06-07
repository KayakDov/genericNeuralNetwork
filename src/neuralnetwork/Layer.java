package neuralnetwork;

import neuralnetwork.ActivationFunctions.ActivationFunction;
import data.Datum;
import java.io.Serializable;
import java.util.function.Function;
import org.jblas.DoubleMatrix;
import org.jblas.NativeBlas;

/**
 * A layer of the neural network.
 * @author Dov Neimand
 */
public class Layer implements Function<DoubleMatrix, DoubleMatrix>, Serializable {

    private DoubleMatrix weights, bias;
    public final ActivationFunction actFunc;
    public final Layer subLayer;
    public final LayerArchitecture architecture;

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
    private Layer(
            DoubleMatrix weights, 
            DoubleMatrix bias, 
            ActivationFunction af, 
            Layer subLayer, 
            LayerArchitecture layerArch) {
        
        this.weights = weights;
        this.bias = bias;
        this.actFunc = af;
        this.subLayer = subLayer;
        architecture = layerArch;
    }


    /**
     * A container for the results of backtracking.
     */
    public class BackTrackResult {

        public final DoubleMatrix grad;
        public final DoubleMatrix val;

        public BackTrackResult(DoubleMatrix grad, DoubleMatrix result) {
            this.grad = grad;
            this.val = result;
        }
    }

    /**
     * Adds the product of a and b to the beginning of c.
     * @param a The first matrix, n x m.
     * @param b The second matrix, m x r.
     * @param c The resulting matrix n x p where p > r.
     */
    private static void gemm(DoubleMatrix a, DoubleMatrix b, DoubleMatrix c) {
        NativeBlas.dgemm('N', 'N', c.rows, b.columns, a.columns, 1, a.data, 0, a.rows, b.data, 0, b.rows, 1, c.data, 0, c.rows);
    }
    
    /**
     * This method calculated the partial derivative of weights and biases that
     * are in the operand. Multiplies the weights by the gradient of the
     * sublayer and copies the results into the gradient on this layer.
     *
     * @param grad The gradient so far calculated for this layer.
     * @param btr The results from the sublayer.
     */
    private void wInOperand(DoubleMatrix grad, BackTrackResult btr) {
        if (hasSubLayer())  gemm(weights, btr.grad, grad);
    }

    /**
     * This is for calculates the the partial derivative with respect to weights
     * in this layer.
     *
     * @param btr The result of previous layers.
     * @param grad The uncompleted gradient.
     * @return The index of the last weight.
     */
    public void wIsWeight(DoubleMatrix grad, BackTrackResult btr) {
        for (int col = 0, w = 0; col < weights.columns; col++) //indecies ordered (0,0), (1,0), ..., (n, 0), (1,0), (1,1), ..., (1,n), ...
            for (int row = 0; row < weights.rows; row++, w++)
                grad.put(row, btr.grad.columns + w, btr.val.get(col));
    }

    /**
     * Computes the partial derivative with respect to the biases in this layer,
     * and adds them to the gradient being built.
     *
     * @param grad The gradient being built.
     * @param btr The result of the sublayer work.
     */
    private void wIsBias(DoubleMatrix grad, BackTrackResult btr) {
        for (int row = 0; row < weights.rows; row++)
            grad.put(row, btr.grad.columns + weights.length + row, 1);
    }
    
    private BackTrackResult subLayerGrad(Datum x){
        return hasSubLayer()
                ? subLayer.grad(x)
                : new BackTrackResult(DoubleMatrix.EMPTY, x);
    }

    /**
     * The gradient of this layer as a function of the weights and biases
     * applied to x.
     *
     * @param x The point the neural network is being applied to. For the
     * purposes of computing the gradient, this is considered a constant.
     * @return The of this method and the result of applying the layer
     * to x. In the gradient vector, each row is the gradient over all the
     * weights and biases for one of the highest layer nodes.
     */
    public BackTrackResult grad(Datum x) {

        BackTrackResult btr = subLayerGrad(x);

        DoubleMatrix grad
                = new DoubleMatrix(architecture.rows, architecture.startIndex + architecture.length());

        wInOperand(grad, btr);
        wIsWeight(grad, btr);
        wIsBias(grad, btr);
        
        ActivationFunction.AtVector actFuncAt = actFunc.ati(affineTransf(btr.val));

        return new BackTrackResult(
                grad.muliColumnVector(actFuncAt.ddt),
                actFuncAt.val
        );
    }

    /**
     * Creates a neural layer from a vector
     *
     * @param vector The vector of all the weights and biases for the neural
     * network.
     * @param ld The architecture for this layer.
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
     * @return A new vector that is the val of the affine transformation
 applied to vec.
     */
    public DoubleMatrix affineTransf(DoubleMatrix vec) {
        return (weights.mmul(vec)).addi(bias);
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix vec) {
        return actFunc.applyi(affineTransf(operand(vec)));
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
