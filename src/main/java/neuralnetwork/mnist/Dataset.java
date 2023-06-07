package neuralnetwork.mnist;

import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public interface Dataset<T> {
    DoubleMatrix getData(int index);
    T getLabel(int index);
    T getPredictedValue(DoubleMatrix m);
    double getConfidence(DoubleMatrix m, T value);
    DoubleMatrix getLabelMatrix(int index);
    int length();
    int numInputs();
    int numOutputs();
    default Dataset<T> combineWith(Dataset<T>... others) {
        CombinedDataset<T> combined =
                this instanceof CombinedDataset<T>
                        ? (CombinedDataset<T>) this
                        : new CombinedDataset<T>();

        return combined.add(others);
    }
    class CombinedDataset<T> implements Dataset<T> {
        private final List<Dataset<T>> datasets = new ArrayList<>();

        @Override
        public DoubleMatrix getData(int index) {
            int sum = 0;
            for (Dataset<T> dataset : datasets) {
                int bound = sum + dataset.length();
                if (index < sum) {
                    return dataset.getData(index - sum);
                }
                sum = bound;
            }
            throw new IndexOutOfBoundsException();
        }

        @Override
        public T getLabel(int index) {
            int sum = 0;
            for (Dataset<T> dataset : datasets) {
                int bound = sum + dataset.length();
                if (index < sum) {
                    return dataset.getLabel(index - sum);
                }
                sum = bound;
            }
            throw new IndexOutOfBoundsException();
        }

        @Override
        public T getPredictedValue(DoubleMatrix m) {
            return datasets.get(0).getPredictedValue(m);
        }

        @Override
        public double getConfidence(DoubleMatrix m, T value) {
            return datasets.get(0).getConfidence(m, value);
        }

        @Override
        public DoubleMatrix getLabelMatrix(int index) {
            return datasets.get(0).getLabelMatrix(index);
        }

        @Override
        public int length() {
            return datasets.get(0).length();
        }

        @Override
        public int numInputs() {
            return datasets.get(0).numInputs();
        }

        @Override
        public int numOutputs() {
            return datasets.get(0).numOutputs();
        }

        public CombinedDataset<T> add(Dataset<T>... others) {
            datasets.addAll(Arrays.asList(others));
            return this;
        }
    }
}
