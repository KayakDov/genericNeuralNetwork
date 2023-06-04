package data;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 * A matrix of integers. The underlying array is row major.
 *
 * @author Dov Neimand
 */
public class MNISTMatrix {

    public final int[] data;
    public final int rows, cols;

    /**
     * How this data is classified, a number between 0 and 9.
     */
    public final int classification;

    /**
     *
     * @param rows The number of rows.
     * @param cols The number of columns.
     * @param data The underlying array in row major format. Changes to this
     * array will effect the matrix and vice versa.
     * @param classification How the matrix is classified.
     *
     */
    public MNISTMatrix(int rows, int cols, int[] data, int classification) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
        this.classification = classification;
    }
    
    /**
     * 
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param data These will get rounded to the nearest integers.
     * @param classification The classification of the data.
     */
    public MNISTMatrix(int rows, int cols, double[] data, int classification) {
        this(
                rows, 
                cols, 
                Arrays.stream(data).mapToInt(i -> (int)Math.round(i)).toArray(), 
                classification
        );
    }

    /**
     * Creates a matrix of all zeroes.
     *
     * @param rows The number of rows.
     * @param cols The number of columns.
     * @param classification The classification of this matrix.
     */
    public MNISTMatrix(int rows, int cols, int classification) {
        this(rows, cols, new int[rows * cols], classification);
    }

    /**
     * Gets an element in the matrix.
     *
     * @param row The row of the element.
     * @param col The column of the element.
     * @return The element at matrix[row][col].
     */
    public final int get(int row, int col) {
        return data[row * cols + col];
    }

    /**
     * Puts a value into the array with indices row, col.
     *
     * @param row The row the value is to be put on.
     * @param col The column the value is to be put on.
     * @param val The value to be put at matrix[row][col].
     */
    public void put(int row, int col, int val) {
        data[row * cols + col] = val;
    }

    public void savePicture(String fileName) {
        try {
            BufferedImage image = new BufferedImage(cols, rows, BufferedImage.TYPE_BYTE_GRAY);
            
            WritableRaster raster = image.getRaster();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    raster.setPixel(j, i, new int[]{get(i,j)});
                }
            }
            
            ImageIO.write(image, "gif", new File(fileName));
        } catch (IOException ex) {
            Logger.getLogger(MNISTMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
