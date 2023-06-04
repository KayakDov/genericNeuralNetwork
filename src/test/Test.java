package test;

import Data.DiskSampleDataSet;
import data.ClassifiedData;
import data.Datum;
import data.MNISTData;
import neuralnetwork.NetworkArchitecture;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuralNetworkBuilder;
import neuralnetwork.ActivationFunctions.Sigmoid;
import optimization.GradDescentBackTrack;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Kayak
 */
public class Test {

    public static DiskSampleDataSet data() {
        int numPointsInSet = 1000;
        return new DiskSampleDataSet(
                new DiskSampleDataSet.Disk[]{
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{0, 0}), 1),
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{0, 2}), 1),
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{2, 0}), 1)
                });
    }

    public static void testMNIST(){
        ClassifiedData trainingData = new MNISTData(true);
        NetworkArchitecture nw = new NetworkArchitecture(new Sigmoid(), trainingData.dim(), 16, 16, 10);//16, 16, 10 for image recognition.
        double[] x = new GradDescentBackTrack(new NeuralNetworkBuilder(trainingData, nw), 1e-5).invoke();
        NeuralNetwork nn = new NeuralNetwork(x, nw);
        
        MNISTData testData = new MNISTData(false);

        double succeed = 0, fail = 0;
        while(testData.hasNext()){
            
            if(nn.correctlyPredicts(testData.next()))
                succeed++;
            else fail++;
        }
            
        System.out.println(succeed/(succeed + fail));
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {//TODO: set up stochastic gradient descent.

        testMNIST();

    }

}
