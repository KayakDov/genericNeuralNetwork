package neuralnetwork.ActivationFunctions;

import java.util.Arrays;
import neuralnetwork.ActivationFunctions.ActivationFunction;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class Sigmoid implements ActivationFunction{

    @Override
    public double ddt(double x) {
        double sig = apply(x);
        return sig*(1-sig);
    }

    @Override
    public double applyAsDouble(double operand) {
        return 1/(1 + Math.exp(-operand));
    }

    @Override
    public AtVector ati(DoubleMatrix x) {
        double[] sig = x.data;
        Arrays.setAll(sig, i -> applyAsDouble(x.data[i]));
        double[] ddt = new double[x.length];
        Arrays.setAll(ddt, i -> sig[i] * (1 - sig[i]));
        return new AtVector(x, new DoubleMatrix(ddt));
    }
    
    

}
