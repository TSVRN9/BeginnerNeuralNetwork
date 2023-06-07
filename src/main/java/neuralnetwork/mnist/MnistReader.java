package neuralnetwork.mnist;

import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.Objects;
import java.util.zip.GZIPInputStream;

public class MnistReader {

    // copied and modified from GitHub
    public static MnistDataset readData(String dataFilePath, String labelFilePath) {
        try {
            DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new GZIPInputStream(Objects.requireNonNull(ClassLoader.getSystemResourceAsStream(dataFilePath)))));

            int magicNumber = dataInputStream.readInt();
            int numberOfItems = dataInputStream.readInt();
            int rows = dataInputStream.readInt();
            int columns = dataInputStream.readInt();

            DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new GZIPInputStream(Objects.requireNonNull(ClassLoader.getSystemResourceAsStream(labelFilePath)))));
            int labelMagicNumber = labelInputStream.readInt();
            int numberOfLabels = labelInputStream.readInt();

            assert numberOfItems == numberOfLabels;

            // int[][][] data = new int[numberOfItems][rows][columns];
            DoubleMatrix[] data = new DoubleMatrix[numberOfItems];
            int[] labels = new int[numberOfLabels];

            for (int i = 0; i < numberOfItems; i++) {
                labels[i] = labelInputStream.readUnsignedByte();
                /* for (int r = 0; r < rows; r++) {
                 *     for (int c = 0; c < columns; c++) {
                 *         data[i][r][c] = dataInputStream.readUnsignedByte();
                 *     }
                 * }
                 */
                DoubleMatrix image = DoubleMatrix.zeros(rows * columns);
                for (int j = 0; j < rows * columns; j++) {
                    image.put(j, dataInputStream.readUnsignedByte() / 255.0);
                }
                data[i] = image;
            }

            dataInputStream.close();
            labelInputStream.close();

            return new MnistDataset(data, labels, rows, columns);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
