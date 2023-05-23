package myneuralnetwork;

import java.util.Arrays;
import java.util.Collections;
import java.util.function.ToDoubleFunction;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 *
 * @author Dov Neimand
 */
public class TrainingDataSet {

    private final Datum[] centeredData, offCenteredData, allClassifiedData;
    private final RVec offSet;
    
    /**
     * True to convey that datum is centered and false otherwise.
     */
    public static final boolean CENTERED = true, LABELED =  true;

    public TrainingDataSet(int dim, int numTruePoints, int numFalsePoints, RVec offset) {
        this.offSet = offset;

        centeredData = new Datum[numTruePoints];
        offCenteredData = new Datum[numFalsePoints];
        allClassifiedData = new Datum[numTruePoints + numFalsePoints];

        Arrays.setAll(centeredData, i -> centeredDatum(LABELED, dim));
        Arrays.setAll(offCenteredData, i -> offCenteredDatum(LABELED, dim));
        System.arraycopy(centeredData, 0,
                allClassifiedData, 0,
                centeredData.length);
        System.arraycopy(offCenteredData, 0,
                allClassifiedData, centeredData.length,
                offCenteredData.length);
        Collections.shuffle(Arrays.asList(allClassifiedData));

        fisherYatesShuffle();
    }

    /**
     * Generates a centered data point.
     *
     * @param labeled True if the centered data point is labeled and false
     * otherwise.
     * @return A centered data point.
     */
    public Datum centeredDatum(boolean labeled, int dim) {
        RVec randVec = RVec.random(dim);
        return labeled ? new Datum(randVec, CENTERED) : new Datum(randVec);
    }

    /**
     * An off centered datum.
     *
     * @param labeled True if the Datum classification is labeled and false
     * otherwise.
     * @param dim the dimension of the datum being generated.
     * @return An off centered datum.
     */
    public Datum offCenteredDatum(boolean labeled, int dim) {
        RVec randVec = RVec.random(dim).plus(offSet);
        return labeled ? new Datum(randVec, !CENTERED): new Datum(randVec);
    }

    /**
     * Shuffles the allClassifiedDatu array.
     */
    private void fisherYatesShuffle() {
        for (int i = 0; i < allClassifiedData.length - 1; i++)
            swapData(
                    i, 
                    i + 1 + RVec.rand.nextInt(allClassifiedData.length - i - 1)
            );
    }

    /**
     * This method swaps two elements in allClassifiedData.
     *
     * @param i The first datum to be swapped.
     * @param j The second datum to be swapped.
     */
    private void swapData(int i, int j) {
        Datum temp = allClassifiedData[i];
        allClassifiedData[i] = allClassifiedData[j];
        allClassifiedData[j] = temp;
    }
    
    /**
     * A parallelStream of all the data in the set.
     * @return 
     */
    public Stream<Datum> parallelStream(){
        return Arrays.stream(allClassifiedData).parallel();
    }
    
    /**
     * Creates a stream that maps each datum to a double.
     * @param f The mapping.
     * @return A double stream mapped from the data.
     */
    public DoubleStream mapToDouble(ToDoubleFunction<Datum> f){
        return parallelStream().mapToDouble(f);
    }
    @Override
    public String toString() {
        return Arrays.toString(allClassifiedData).replace("), (", ")\n(").replace("]", "").replace("[", "");
    }
    
    

}
