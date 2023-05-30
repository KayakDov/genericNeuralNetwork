package data;

import java.util.stream.Stream;

/**
 * An object that can provide a stream of classified data.
 * @author Dov Neimand
 */
public interface ClassifiedData {

    /**
     * A stream of all the data.
     * @return A stream of all the data.
     */
    public abstract Stream<Datum> stream();
    
    /**
     * A stream of randomly selected elements.
     * @param size The number of elements in the stream.
     * @return A stream of randomly selected elements.
     */
    public abstract Stream<Datum> stochasticStream(int size);

    /**
     * A parallel stream of data.
     * @return A parallel stream of data.
     */
    public default Stream<Datum> parallel() {
        return stream().parallel();
    }
    
    /**
     * A parallel stream of randomly selected elements.
     * @param size The number of elements in the desired stream.
     * @return A parallel stream of randomly selected data.
     */
    public default Stream<Datum> stochasticParallel(int size) {
        return stochasticStream(size).parallel();
    }

    /**
     * The number of classifications available for data.
     * @return The number of classifications available for data.
     */
    public default int numTypes() {
        return stream().findAny().get().numTypes;
    }
    
    /**
     * The dimension of the data space.
     * @return The dimension of the data space.
     */
    public default int dim(){
        return stream().findAny().get().length;
    }
}
