
package test;

import Data.Datum;
import Data.DiskSampleDataSet;
import java.util.Arrays;
import java.util.List;
import neuralnetwork.NetworkArchitecture;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuralNetworkBuilder;
import neuralnetwork.Sigmoid;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Kayak
 */
public class Test {

    public static List<Datum> data(){
        int numPointsInSet = 1000;
        return new DiskSampleDataSet(
                new DiskSampleDataSet.Disk[]{
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{0,0}), 1), 
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{0, 2}), 1)
                });
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        NeuralNetworkBuilder nnb = new NeuralNetworkBuilder(
                data(), 
                new NetworkArchitecture(new Sigmoid(), 2, 2),
                1e-5
        );
        
        NeuralNetwork nn = nnb.compute();
        
        System.out.println(nn.apply(0.0, 0));
        System.out.println(nn.apply(0, 1));
        System.out.println(nn.apply(0, 2));

        
    }
    
}
