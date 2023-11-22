package neuralnetwork;

import data.Datum;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.function.Function;
import optimization.FuncAt;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class NeuralNetwork implements Function<DoubleMatrix, DoubleMatrix>, Serializable {

    private Layer topLayer;

    /**
     * The networks architecture.
     */
    public final Architecture architecture;

    /**
     * Creates a neural network from a vector in Rn and information about each
     * layer's size.
     *
     * @param x The vector he network is generated from.
     * @param layerDims A description of each layer's size.
     */
    public NeuralNetwork(double[] x, Architecture layerDims) {
        architecture = layerDims;
        for (int layerInd = 0; layerInd < layerDims.numLayers(); layerInd++)
            topLayer = new Layer(x, layerDims.get(layerInd), layerDims.getActFunc(), topLayer);
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix x) {
        return topLayer.apply(x);
    }
    
    /**
     * This method attempts to classify x;
     * @param x Some datum.
     * @return The predicted classification of x.
     */
    public int prediction(DoubleMatrix x){
        return apply(x).argmax();
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
     * QUickly transposes a vector in place.
     * @param vec The vector to be transposed.
     */
    private static DoubleMatrix transpose(DoubleMatrix vec){
        vec.rows = 1;
        vec.columns = vec.length;
        return vec;
    }
    /**
     * The gradient of the neural network relative to the weights and biases at
     * x.
     *
     * @param x The datum for which the gradient is calculated.
     * @return The gradient of the cost.
     */
    public FuncAt gradCost(Datum x) {
        Layer.BackTrackResult nnResult = topLayer.grad(x);
        nnResult.val.data[x.type] -= 1;
        DoubleMatrix gradCost = transpose(nnResult.val).mmul(nnResult.grad);
        return new FuncAt(gradCost, nnResult.val.dot(nnResult.val));
    }

    /**
     * How accurate is the neural networks prediction for the proffered datum.
     *
     * @param x A datum, presumably in the training set.
     * @return A measure of how accurate the nerual networks result is on x. A
     * high number means the network did a bad job at classifying the data, and
     * a number close to 0 is a good job.
     *
     * (nn(x) - x.type)*(nn(x) - x.type)
     */
    public double cost(Datum x) {
        DoubleMatrix forecast = apply(x);
        forecast.data[x.type] -= 1;
        return forecast.dot(forecast);
    }

    /**
     * Does this neural network give the correct result for the datum.
     * @param x The datum being checked.
     * @return True if the network yields the correct result, false otherwise.
     */
    public boolean correctlyPredicts(Datum x){
        return apply(x).argmax() == x.type; 
    }
    
    /**
     * Saves this neural network to a file.
     * @param fileName The name of the file.
     * @throws java.io.FileNotFoundException
     * @throws java.io.IOException
     */
    public void saveToFile(String fileName) 
            throws FileNotFoundException, IOException{
        new ObjectOutputStream(new FileOutputStream(fileName))
                .writeObject(this);
    }

    /**
     * Constructs a saves neural network from a file.
     * @param fileName The name of the file.
     * @return A neural network that was saved to a file.
     * @throws java.io.FileNotFoundException
     * @throws java.lang.ClassNotFoundException
     */
    public static NeuralNetwork fromFile(String fileName) 
            throws FileNotFoundException, IOException, ClassNotFoundException{
        return (NeuralNetwork)new ObjectInputStream(
                new FileInputStream(fileName)).readObject();
    }
    
    
   /**
    * Runs some basic tests on this class.
    * @param args Not used.
    */
    public static void main(String[] args) {
        DoubleMatrix id = new DoubleMatrix(2, 2, 1, 0, 0, 1);
        DoubleMatrix m = new DoubleMatrix(2, 2, 1, 2, 3, 4);//TODO: Check every data access!
        System.out.println(m.get(0, 1));
    }

}
