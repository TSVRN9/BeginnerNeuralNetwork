package neuralnetwork.networks;

import neuralnetwork.NetworkVector;
import neuralnetwork.NeuralNetwork;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SigmoidSquaredNetwork extends NeuralNetwork {
    public SigmoidSquaredNetwork(int[] layerSizes) {
        super(layerSizes);
    }

    public SigmoidSquaredNetwork(double[][][] weights, double[][] biases, int[] layerSizes) {
        super(weights, biases, layerSizes);
    }

    public SigmoidSquaredNetwork(NetworkVector vector) {
        super(vector);
    }

    @Override
    public DoubleMatrix activation(DoubleMatrix m) {
        DoubleMatrix ones = DoubleMatrix.ones(m.rows, m.columns);
        return ones.div(ones.add(MatrixFunctions.exp(m.neg())));
    }

    @Override
    public DoubleMatrix activationPrime(DoubleMatrix m) {
        // sigmoid(m)(1 - sigmoid(m))
        DoubleMatrix result = activation(m);
        return result.mul(result.sub(1).mul(-1));
    }

    @Override
    public DoubleMatrix cost(DoubleMatrix m, DoubleMatrix y) {
        // (m - y) ^ 2
        return MatrixFunctions.pow(m.sub(y), 2);
    }

    @Override
    public DoubleMatrix costPrime(DoubleMatrix m, DoubleMatrix y) {
        // 2 * (m - y)
        return m.sub(y).mul(2);
    }
}
