package neuralnetwork.mnist;

import org.jblas.DoubleMatrix;

public class XorDataset implements Dataset<Boolean> {
    @Override
    public DoubleMatrix getData(int index) {
        boolean a = index - 1 >= 0;
        boolean b = index - 2 >= 0;

        return new DoubleMatrix(new double[] { a ? 1 : 0, b ? 1 : 0 });
    }

    @Override
    public Boolean getLabel(int index) {
        boolean a = index - 1 >= 0;
        boolean b = index - 2 >= 0;

        return Boolean.logicalXor(a, b);
    }

    @Override
    public Boolean getPredictedValue(DoubleMatrix m) {
        return m.get(0) >= .5;
    }

    @Override
    public double getConfidence(DoubleMatrix m, Boolean value) {
        return value ? 1 - m.get(0) : m.get(0);
    }

    @Override
    public DoubleMatrix getLabelMatrix(int index) {
        return booleanToMatrix(getLabel(index));
    }

    @Override
    public int length() {
        return 4;
    }

    private DoubleMatrix booleanToMatrix(boolean bool) {
        return bool ? DoubleMatrix.ones(1) : DoubleMatrix.zeros(1);
    }

    @Override
    public int numInputs() {
        return 2;
    }

    @Override
    public int numOutputs() {
        return 1;
    }
}
