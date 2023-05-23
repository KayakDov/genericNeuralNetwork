package myneuralnetwork;

/**
 * This class is meant to be a sandbox to test the neural network.
 * @author Dov Neimnad
 */
public class TestNeuralNetwork {

    /**
     * The main function tests the neural network.
     * @param args There are no command line arguments.
     */
    public static void main(String[] args) {
        TrainingDataSet tds = new TrainingDataSet(2, 1000, 1000, new RVec(1.5, 0));
        
        NeuralNode nn = new NeuralNode(tds, 1e-8);

        System.out.println(nn.toString().replace(",", "*x + ").replace("(", "").replace(")*x", "*y"));
    }

}
