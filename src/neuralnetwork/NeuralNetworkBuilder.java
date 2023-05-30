package neuralnetwork;

import java.util.concurrent.RecursiveTask;
import optimization.DiffReal;
import optimization.GradDescentBackTrack;
import org.jblas.DoubleMatrix;
import data.ClassifiedData;
import optimization.FuncAt;

/**
 * This class optimizes a neural network for given data.
 *
 * @author Dov Neimand
 */
public class NeuralNetworkBuilder extends RecursiveTask<NeuralNetwork> {

    private final ClassifiedData trainingData;
    private final NetworkArchitecture layerDims;
    private final double threshold;

    /**
     * TODO: pass network architecture as single variable?
     * @param data The data used to train the network.
     * @param architecture The number of nodes in each layer.
     * @param af The activation function used by each node.
     * @param A greater than 0 value used for the optimization
     */
    public NeuralNetworkBuilder(ClassifiedData data, NetworkArchitecture architecture, double threshold) {
        if (data.numTypes() != architecture.outputDim())
            throw new IllegalArgumentException("The number of output dimensions, "
                    + architecture.outputDim() + " must match the number of data "
                    + "types, " + data.numTypes());
        
        if(data.dim() != architecture.inputDim())
            throw new IllegalArgumentException("The data passed has dimesnion " 
                    + data.dim() + " but the architecure calls for data with "
                            + "dimesnion " + architecture.inputDim());
        
        this.trainingData = data;
        this.layerDims = architecture;
        this.threshold = threshold;

    }


    /**
     *
     * @param nn a Neural Network.
     * @return The cost of the neural network over the given data set.
     */
    private double cost(NeuralNetwork nn) {
        return trainingData.parallel().mapToDouble(datum -> nn.cost(datum)).sum();
    }
    
    
    /**
     * The change in the cost function as the weights and biases are changed.
     *
     * @param nn The current set of weights and biases.
     * @param stochasticSize How many elements of data should be considered? 
     * Pass -1 to consider all of them.
     * @return The gradient of the neural network as a function of its weights
     * and biases.
     */
    private FuncAt gradCost(NeuralNetwork nn, int stochasticSize) {
        
        return (stochasticSize != -1? 
                trainingData.stochasticParallel(stochasticSize):
                trainingData.parallel())
                .map(x -> nn.gradCost(x))
                .collect(
                    () -> new FuncAt(
                            new DoubleMatrix(1, nn.numWeightsAndBiases()), 
                            0
                    ), 
                    (a,b) -> a.addi(b), 
                    (a,b) -> a.addi(b)
                );
    }

    @Override
    public NeuralNetwork compute() {

        DiffReal nnBuilderDiffReal = new DiffReal() {
            @Override
            public DoubleMatrix grad(double[] x) {
                return funcAt(x).grad;
            }

            @Override
            public int domainDim() {
                return layerDims.numVariables();
            }

            @Override
            public double applyAsDouble(double[] value) {
                return cost(new NeuralNetwork(value, layerDims));
            }

            @Override
            public FuncAt funcAt(double[] x) {
                return gradCost(new NeuralNetwork(x, layerDims));
            }
            
            
        };

        return new NeuralNetwork(
            new GradDescentBackTrack(nnBuilderDiffReal, threshold
            ).compute(), layerDims);
    }

}
