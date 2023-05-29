package data;

import java.util.stream.Stream;

/**
 * An object that can provide a stream of classified data.
 * @author Dov Neimand
 */
public interface ClassifiedData {

    public abstract Stream<Datum> stream();

    public default Stream<Datum> parallel() {
        return stream().parallel();
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
