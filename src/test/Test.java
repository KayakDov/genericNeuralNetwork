package test;

import Data.DiskSampleDataSet;
import data.ClassifiedData;
import data.MNISTData;
import neuralnetwork.Architecture;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuralNetworkBuilder;
import neuralnetwork.ActivationFunctions.Sigmoid;
import optimization.GradDescentBackTrack;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
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

    public static void simpleTest() {
        ClassifiedData data = data();
        Architecture nw = new Architecture(new Sigmoid(), data.dim(), 3, 3);//16, 16, 10 for image recognition.
        double[] x = new GradDescentBackTrack(new NeuralNetworkBuilder(data(), nw), 1e-12).invoke();

        NeuralNetwork nn = new NeuralNetwork(x, nw);

        System.out.println(nn.apply(0.0, 0));
        System.out.println(nn.apply(0, 1));
        System.out.println(nn.apply(0, 2));
        System.out.println(nn.apply(2, 0));
    }

    public static void MNIST() {
        ClassifiedData data = new MNISTData(true);
        
        Architecture nw = new Architecture(new Sigmoid(), data.dim(), 100, 50, 10);//16, 16, 10 for image recognition.
        double[] x = new GradDescentBackTrack(new NeuralNetworkBuilder(data, nw), 1e-4).invoke();

        NeuralNetwork nn = new NeuralNetwork(x, nw);

        MNISTData testSet = new MNISTData(false);

        long correct = testSet.parallel().filter(datum -> nn.correctlyPredicts(datum)).count();

        System.out.println((double) correct / testSet.size());
    }

    

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {//TODO: set up stochastic gradient descent.
        MNIST();
//        simpleTest();

//        DoubleMatrix a = new DoubleMatrix(2, 2, new double[]{1,2,3,4});
//        DoubleMatrix b = new DoubleMatrix(2, 2, new double[]{1,0,0,1});
//        DoubleMatrix c = new DoubleMatrix(4, 4);
//        
//        c.rows = c.columns = 2;
//        
//        a.mmuli(b, c);
//        
//        System.out.println(c);
//        System.out.println(c.data.length);

    }

}
