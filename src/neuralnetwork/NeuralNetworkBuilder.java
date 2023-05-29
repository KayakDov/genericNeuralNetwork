package neuralnetwork;

import data.Datum;
import java.util.List;
import java.util.concurrent.RecursiveTask;
import optimization.DiffReal;
import optimization.GradDescentBackTrack;
import org.jblas.DoubleMatrix;
import data.ClassifiedData;

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
            throw new RuntimeException("The number of output dimensions, "
                    + architecture.outputDim() + " must match the number of data "
                    + "types, " + data.numTypes());
        this.trainingData = data;
        this.layerDims = architecture;
        this.threshold = threshold;
//        System.out.println("neuralnetwork.NeuralNetworkBuilder.<init>()");
//        System.out.println(data.toString().replace("[", "(").replace("]", ")").replace(";", ",").replace("), (", ")\n("));
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
     * @return The gradient of the neural network as a function of its weights
     * and biases.
     */
    private DoubleMatrix gradCost(NeuralNetwork nn) {
        
        return trainingData.parallel().map(x -> nn.gradCost(x))
                .collect(
                () -> new DoubleMatrix(1, nn.numWeightsAndBiases()), 
                (a,b) -> a.addi(b), 
                (a,b) -> a.addi(b));
        
//        double[] dCostdw = new double[nn.numWeightsAndBiases()];
//        Arrays.parallelSetAll(dCostdw, i -> trainingData.stream().mapToDouble(x -> dCostdw(nn, x, i)).sum());
//        return new DoubleMatrix(dCostdw);
    }

    @Override
    public NeuralNetwork compute() {

        DiffReal nnBuilderDiffReal = new DiffReal() {
            @Override
            public DoubleMatrix grad(double[] x) {
                return gradCost(new NeuralNetwork(x, layerDims));
            }

            @Override
            public int domainDim() {
                return layerDims.numVariables();
            }

            @Override
            public double applyAsDouble(double[] value) {
                return cost(new NeuralNetwork(value, layerDims));
            }
        };

        return new NeuralNetwork(
            new GradDescentBackTrack(nnBuilderDiffReal, threshold
            ).compute(), layerDims);
    }

}
