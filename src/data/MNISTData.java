package data;

import java.io.*;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Dov Neimand
 */
public class MNISTData implements Iterator<MNISTDatum>{

    public static final String dataFilePath = "MNISTData\\train-images-idx3-ubyte",
            labelFilePath = "MNISTData\\train-labels-idx1-ubyte";

    private DataInputStream labelInputStream, dataInputStream;

    private int rows, cols, size, labelSize;
    public Datum[] data;

    int index = 0;

    private byte[] getMagicNumber(DataInputStream dis) throws IOException {
        dis.readByte();
        dis.readByte();
        return new byte[]{dis.readByte(), dis.readByte()};
    }

    public MNISTData() {

        try {
            dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
            labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));

            getMagicNumber(dataInputStream);
            getMagicNumber(labelInputStream);

            size = dataInputStream.readInt();
            rows = dataInputStream.readInt();
            cols = dataInputStream.readInt();

        } catch (FileNotFoundException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public boolean hasNext() {
        return index < size;
    }

    @Override
    public MNISTDatum next() {
        try {
            int label = labelInputStream.readUnsignedByte();
            if(!hasNext()) close();
            return new MNISTDatum(nextMatrix().data, label);
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
            throw new RuntimeException();
        }
    }
    
    /**
     * A stream of all the datum.
     * @return A stream of all the datum.
     */
    public Stream<Datum> stream(){
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(this, Spliterator.ORDERED),
          false);
    }
    
    /**
     * Each element in the data has a relativeSize chance of making it into
     * the stream.
     * @param relativeSize The probability that a given element is in the 
     * stream.
     * @return A stream cut down from the original.
     */
    public Stream<Datum> stochasticStream(double relativeSize){
        return stream().filter(datum -> Math.random() < relativeSize);
    }

    /**
     * The next matrix in the set.
     * @return The next matrix in the set.
     */
    public DoubleMatrix nextMatrix() {

        try {
            DoubleMatrix val = new DoubleMatrix(rows, cols);
            
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    val.put(r, c, dataInputStream.readUnsignedByte());
            
            index++;
            return val;
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
            throw new RuntimeException();
        }
    }

    public static void main(String[] args) throws IOException {
        MNISTData test = new MNISTData();
        System.out.println(test.stream().count());
    }

    public void close() {
        try {
            labelInputStream.close();
            dataInputStream.close();
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}

//http://yann.lecun.com/exdb/mnist/#:~:text=The%20magic%20number%20is%20an,2%20bytes%20are%20always%200.&text=The%204%2Dth%20byte%20codes,most%20non%2DIntel%20processors).
//THE IDX FILE FORMAT
//the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
//The basic format is
//
//magic number
//size in dimension 0
//size in dimension 1
//size in dimension 2
//.....
//size in dimension N
//data
//
//The magic number is an integer (MSB first). The first 2 bytes are always 0.
//
//The third byte codes the type of the data:
//0x08: unsigned byte
//0x09: signed byte
//0x0B: short (2 bytes)
//0x0C: int (4 bytes)
//0x0D: float (4 bytes)
//0x0E: double (8 bytes)
//
//The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
//
//The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
//
//The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
