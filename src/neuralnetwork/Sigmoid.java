package neuralnetwork;

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

}
