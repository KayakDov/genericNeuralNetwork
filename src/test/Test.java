
package test;

import data.Datum;
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

    public static DiskSampleDataSet data(){
        int numPointsInSet = 1000;
        return new DiskSampleDataSet(
                new DiskSampleDataSet.Disk[]{
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{0,0}), 1), 
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{0, 2}), 1),
                    new DiskSampleDataSet.Disk(numPointsInSet, new DoubleMatrix(new double[]{2, 0}), 1)
                });
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        NetworkArchitecture nw = new NetworkArchitecture(new Sigmoid(), 2, 3, 3);
                        
        NeuralNetworkBuilder nnb = new NeuralNetworkBuilder(
                data(), 
                nw,
                1e-6
        );
        
        NeuralNetwork nn = nnb.compute();
        
        System.out.println(nn.apply(0.0, 0));
        System.out.println(nn.apply(0, 1));
        System.out.println(nn.apply(0, 2));
        System.out.println(nn.apply(2, 0));

        
    }
    
}
