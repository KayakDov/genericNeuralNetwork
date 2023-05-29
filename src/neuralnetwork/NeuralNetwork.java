package neuralnetwork;

import data.Datum;
import java.util.Arrays;
import java.util.function.Function;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class NeuralNetwork implements Function<DoubleMatrix, DoubleMatrix> {

    private Layer topLayer;

    /**
     * The networks arch.
     */
    public final NetworkArchitecture architecture;

    /**
     * Creates a neural network from a vector in Rn and information about each
     * layer's size.
     *
     * @param x The vector he network is generated from.
     * @param layerDims A description of each layer's size.
     */
    public NeuralNetwork(double[] x, NetworkArchitecture layerDims) {
        architecture = layerDims;
        for (int layerInd = 0; layerInd < layerDims.numLayers(); layerInd++)
            topLayer = new Layer(x, layerDims.get(layerInd), layerDims.getActFunc(), topLayer);
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix x) {
        return topLayer.apply(x);
    }

    /**
     * Applies this neural network to the given variable.
     *
     * @param x A datum of unknown classification.
     * @return A prediciton for the classification of the datum.
     */
    public DoubleMatrix apply(double... x) {
        return apply(new DoubleMatrix(x));
    }

    /**
     * Yields the result of one node in the outer most layer of the network
     * applied to the given data.
     *
     * @param x The datum.
     * @param node The index of the desired node.
     * @return The value of the node when applied to the datum.
     */
    public double applyNode(DoubleMatrix x, int node) {
        return topLayer.applyNode(x, node);
    }

    /**
     * The number of output values of the neural network.
     *
     * @return The number of output values of the neural network.
     */
    public int rangeDim() {
        return topLayer.numNodes();
    }

    @Override
    public String toString() {
        StringBuilder toString = new StringBuilder();
        Layer layer = topLayer;
        do {
            toString.append(layer.toString());
            layer = layer.subLayer;
        } while (layer != null);
        return toString.toString();
    }

    /**
     * The total number of weights and biases among all the layers.
     *
     * @return The total number of weights and biases among all the layers.
     */
    public int numWeightsAndBiases() {
        int numWandB = 0;
        Layer layer = topLayer;
        while (layer != null) {
            numWandB += layer.numberWeightsAndBiases();
            layer = layer.subLayer;
        }
        return numWandB;
    }

    /**
     * The gradient of the neural network relative to the weights and biases at
     * x.
     *
     * @param x The datum for which the gradient is calculated.
     * @return The gradient of the cost.
     */
    public DoubleMatrix gradCost(Datum x) {
        Layer.BackTrackResult btr = topLayer.grad(x);
        return (btr.apply.subi(x.type())).transpose().mmul(btr.grad);
    }

    /**
     * How accurate is the neural networks prediction for the proffered datum.
     *
     * @param nn The neural network
     * @param x A datum, presumably in the training set.
     * @return A measure of how accurate the nerual networks result is on x. A
     * high number means the network did a bad job at classifying the data, and
     * a number close to 0 is a good job.
     *
     * (nn(x) - x.type)*(nn(x) - x.type)
     */
    public double cost(Datum x) {
        DoubleMatrix forecast = apply(x);
        forecast = forecast.sub(x.type());
        return forecast.dot(forecast);
    }

    public static void main(String[] args) {
        DoubleMatrix id = new DoubleMatrix(2, 2, 1, 0, 0, 1);
        DoubleMatrix m = new DoubleMatrix(2, 2, 1, 2, 3, 4);//TODO: Check every data access!
        System.out.println(m.get(0, 1));
    }

}
