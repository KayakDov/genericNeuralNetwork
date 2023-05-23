package singleNode;

import optimization.GradDescentBackTrack;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;
import static singleNode.NeuralNode.sigmoid;
import static java.lang.Math.abs;

/**
 *
 * @author Dov Neimand
 */
public class NeuralNode implements ToDoubleFunction<RVec>{

    private final RVec weights;
    private final double bias;
    
    public NeuralNode (TrainingDataSet tds, double tolerance){
        RVec preNN = new NeuralNodeBuilder(tds).getNode(tolerance);
        weights = preNN.subVector(0, preNN.dim() - 1);
        bias = preNN.at(preNN.dim() - 1);
    }

    /**
     * The Constructor.
     *
     * @param weights the weights applied to any vector.
     * @param bias the bias made by the node.
     */
    public NeuralNode(RVec weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    /**
     * The sigmoid value of x see
     * <a href="https://en.wikipedia.org/wiki/Sigmoid_function">wikipedia</a>
     *
     * @param x A value in the domain of the sigmoid function.
     * @return sigmoid(x).
     */
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Runs the given datum through this node.
     *
     * @param x the point to be evaluated.
     * @return A number between 0 and 1. Hopefully loser to one for True datum
     * and 0 for false datum.
     *
     */
    @Override
    public double applyAsDouble(RVec x) {
        return sigmoid(x.dot(weights) + bias);
    }

    @Override
    public String toString() {
        return weights.toString() + "*x + " + bias;
    }
    
    
}

class NeuralNodeBuilder implements DiffReal {

    private final TrainingDataSet tds;

    /**
     * The constructor.
     * @param tds The training data set that the generated node will be 
     * optimized for.
     */
    public NeuralNodeBuilder(TrainingDataSet tds) {
        this.tds = tds;
    }

    
    /**
     * The derivative of the sigmoid function at x.
     * @param x A real number.
     * @return The derivative of sigmoid at x.
     */
    private static double dSdx(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }
    
    /**
     * Finds w*d+b where w is the weights and b the bias.
     * @param weightsAndBias All but the last value are weights. The last values
     * is the bias.
     * @param datum The d value.
     * @return w*d+b
     */
    private static double affTrans(RVec weightsAndBias, Datum datum){
        return IntStream.range(0, datum.dim())
                .mapToDouble(j -> datum.at(j)*weightsAndBias.at(j))
                .sum() + weightsAndBias.at(weightsAndBias.dim() - 1);
    }
    
    /**
     * Applies the potential node defined by the weights and bias vector to
     * the datum.
     * @param weightsAndBias The initial values are the weights and the last 
     * value is the bias.
     * @param datum The datum the potential node is applied to.
     * @return The result of the node, as defined by weightsAndBias applied
     * to the datum.
     */
    private static double applyNodeToDatum(RVec weightsAndBias, Datum datum){
        return sigmoid(affTrans(weightsAndBias, datum));
    }
    
    /**
     * The error Squared for a specific datum with a known classification.
     *
     * @param datum The errorSq of this node for the given classified datum. If
     * the datum is not classified, and exception is thrown.
     * @return
     */
    private double errorSq(RVec weightsAndBias, Datum datum) {
        if (!datum.isClassified()) throw new RuntimeException("The error for "
                    + "unclasified data is unknown.");

        double dif = applyNodeToDatum(weightsAndBias, datum) - datum.getType();

        return dif * dif;
    }

    /**
     * How accurate are the weights of this node for the proffered data set.
     *
     * @param tsData A set to test the node on.
     * @return The square of the difference of these predictions from the actual
     * classifications.
     */
    private double errorSq(RVec weightsAndBias) {
        return tds.mapToDouble(datum -> errorSq(weightsAndBias, datum))
                .average().orElseThrow();
    }

    /**
     * The gradient of the cost/error function of this node as a function of the
     * weights of the node.
     *
     * @param tsData The data set for which this node is being evaluated.
     * @return Te gradient of this nodes error when applied to the training data
     * set.
     */
    private RVec gradErrorSq(RVec weightsAndBias) {
        return new RVec(weightsAndBias.dim(), i -> tds.mapToDouble(datum -> 
                dErrorSqdwi(weightsAndBias, i, datum)).average().orElseThrow());
    }

    /**
     * The derivative of applyAsDouble as function a of weight_i.
     *
     * @param x The datum for which the gradient, as a function of the weights,
     * is calculated.
     * @param i the index of the weight in weights that is differentiated by.
     * @return The derivative for x as a function of weight_i.
     */
    private double dErrorSqdwi(RVec weightsAndBias, int i, Datum x) {
        
        double sigmAffTrans = applyNodeToDatum(weightsAndBias, x);
        
        return 2*(sigmAffTrans - x.getType())
                *(sigmAffTrans*(1-sigmAffTrans))
                *(i < x.dim()?x.at(i):1);
                
//                dSdx(err*err) * 2 * abs(err) 
//                * (err < 0? -1:1) * (i < x.dim()?x.at(i):1);
    }

    @Override
    public RVec grad(RVec x) {
        return gradErrorSq(x);
    }

    @Override
    public int domainDim() {
        return tds.parallelStream().findAny().orElseThrow().dim() + 1;
    }

    @Override
    public double applyAsDouble(RVec value) {
        return errorSq(value);
    }
    
    /**
     * Finds an optimal neural node for the training data set.  The values 
     * up to but not including the last are meant to be the weights, and the
     * last value is the bias.
     * @return An optimal neural node for the training data set.
     */
    public RVec getNode(double tolerance){
        return new GradDescentBackTrack(this, tolerance).run();
    }
}