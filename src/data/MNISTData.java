package data;

import java.io.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

/**
 *
 * @author Dov Neimand
 */
public class MNISTData implements Iterator<MNISTDatum>, ClassifiedData {

    public static final String dataFilePath = "MNISTData\\train-images-idx3-ubyte",
            labelFilePath = "MNISTData\\train-labels-idx1-ubyte",
            dataTestFilePath = "MNISTData\\t10k-images-idx3-ubyte",
            labelTestFilePath = "MNISTData\\t10k-labels-idx1-ubyte";

    private DataInputStream labelReader, dataReader;

    private int rows, cols, size, labelSize;
    public Datum[] data;

    private final boolean bigSet;

    int index = 0;

    private byte[] getMagicNumber(DataInputStream dis) throws IOException {
        dis.readByte();
        dis.readByte();
        return new byte[]{dis.readByte(), dis.readByte()};
    }

    public final void resetFileReader() {
        try {
            if (dataReader != null) close();
            dataReader = new DataInputStream(new BufferedInputStream(new FileInputStream(bigSet ? dataFilePath : dataTestFilePath)));
            labelReader = new DataInputStream(new BufferedInputStream(new FileInputStream(bigSet ? labelFilePath : labelTestFilePath)));

            getMagicNumber(dataReader);
            getMagicNumber(labelReader);

            size = dataReader.readInt();
            rows = dataReader.readInt();
            cols = dataReader.readInt();
            
            for(int i = 0; i < 4; i++) labelReader.read();
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    /**
     *
     * @param bigSet True to use the larger of the two data sets, false for the
     * smaller. The larger should be used for training and the smaller for
     * learning.
     */
    public MNISTData(boolean bigSet) {
        this.bigSet = bigSet;
        resetFileReader();

        data = new Datum[size];
        for (int i = 0; i < data.length; i++)
            data[i] = next();
    }

    @Override
    public boolean hasNext() {
        return index < size;
    }

    @Override
    public MNISTDatum next() {

        try {
            int label = labelReader.readUnsignedByte();
            if (!hasNext()) close();
            double[] ndr = new double[rows * cols];
            for (int i = 0; i < ndr.length; i++)
                ndr[i] = dataReader.readUnsignedByte();
            index++;
            return new MNISTDatum(ndr, label);
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
            throw new RuntimeException(ex);
        }

    }

    /**
     * A stream of all the datum.
     *
     * @return A stream of all the datum.
     */
    @Override
    public Stream<Datum> stream() {
        return Arrays.stream(data);
    }

    /**
     * The next matrix in the set. Warning, the label index must be advanced as
     * well if they're to be kept in sinc.
     *
     * @return The next matrix in the set.
     */
    public MNISTMatrix nextMatrix() {

        try {
            MNISTMatrix val = new MNISTMatrix(rows, cols, labelReader.readUnsignedByte());

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    val.put(c, r, dataReader.readUnsignedByte());

            index++;
            return val;
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
            throw new RuntimeException();
        }
    }

    /**
     * The dimension of the data.
     *
     * @return The dimension of the data.
     */
    public int dataDim() {
        return rows * cols;
    }

    public static void main(String[] args) throws IOException {

        MNISTData test = new MNISTData(false);
        for (int i = 0; i < 10; i++) {
            MNISTMatrix m = new MNISTMatrix(test.rows, test.cols, test.data[i].data, test.data[i].type);
            m.savePicture(m.classification + "_" + i + ".gif");
            System.out.println(m.classification);
        }
    }

    public void close() {
        try {
            labelReader.close();
            dataReader.close();
        } catch (IOException ex) {
            Logger.getLogger(MNISTData.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Datum[] array() {
        return data;
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
