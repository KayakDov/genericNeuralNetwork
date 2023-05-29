package Data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import org.jblas.DoubleMatrix;

/**
 * A set of data assigned to different disks.
 *
 * @author Dov Neimand
 */
public class DiskSampleDataSet extends ArrayList<Datum> {

    /**
     * The basic attributes of a set of points randomly distributed across a
     * disk.
     */
    public static class Disk {

        public final int numPoints;
        public final DoubleMatrix center;
        public final double radius;

        /**
         *
         * @param numPoints The number of points in the disk.
         * @param center The center of the disk.
         * @param radius The radius of the disk.
         */
        public Disk(int numPoints, DoubleMatrix center, double radius) {
            this.numPoints = numPoints;
            this.center = center;
            this.radius = radius;
        }

        /**
         * Generated numVectors uniformly random vectors in a disk centered at
         * center.
         * @param id The id number of the points in this disk.  The number 
         * should not exceed the number of sets being created.
         * @return A list of generated vectors.
         */
        public List<Datum> vectors(int id) {
            return IntStream.range(0, numPoints).mapToObj(i -> {
                DoubleMatrix vec = DoubleMatrix.randn(center.length);
                vec = vec.div(vec.norm2()).mul(radius * Math.random()).add(center);
                return new Datum(vec.data, id);
            }).toList();
        }

    }

    /**
     * Creates a bunch of disk points, each disk having its own id.
     * @param disks The disks in which the points are to be created.
     */
    public DiskSampleDataSet(Disk[] disks) {
        super(Arrays.stream(disks).mapToInt(disk -> disk.numPoints).sum());
        IntStream.range(0, disks.length).forEach(i -> addAll(disks[i].vectors(i)));
    }

}
