package neuralnetwork.mnist;

import org.jblas.DoubleMatrix;

public class MnistDataset implements Dataset<Integer> {
    private final DoubleMatrix[] data;
    private final int[] labels;
    public final int rows, columns;
    private static final DoubleMatrix[] labelToDoubleMatrix = initLabelToDoubleMatrix();

    private static DoubleMatrix[] initLabelToDoubleMatrix() {
        DoubleMatrix[] doubleMatrices = new DoubleMatrix[10];

        for (int i = 0; i < 10; i++) {
            doubleMatrices[i] = DoubleMatrix.zeros(10).put(i, 1);
        }

        return doubleMatrices;
    }

    public MnistDataset(DoubleMatrix[] data, int[] labels, int rows, int columns) {
        this.data = data;
        this.labels = labels;
        this.rows = rows;
        this.columns = columns;
    }

    @Override
    public DoubleMatrix getData(int index) {
        return data[index];
    }

    @Override
    public Integer getLabel(int index) {
        return labels[index];
    }

    @Override
    public Integer getPredictedValue(DoubleMatrix m) {
        return m.argmax();
    }

    @Override
    public double getConfidence(DoubleMatrix m, Integer value) {
        return m.get(value);
    }

    public DoubleMatrix getLabelMatrix(int index) {
        return labelToDoubleMatrix[labels[index]];
    }

    @Override
    public int length() {
        return data.length;
    }

    @Override
    public int numInputs() {
        return rows*columns;
    }

    @Override
    public int numOutputs() {
        return 10;
    }
}
