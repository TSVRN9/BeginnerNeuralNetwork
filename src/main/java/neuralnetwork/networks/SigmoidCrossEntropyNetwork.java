package neuralnetwork.networks;

import neuralnetwork.NetworkVector;
import neuralnetwork.NeuralNetwork;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SigmoidCrossEntropyNetwork extends NeuralNetwork {
    public SigmoidCrossEntropyNetwork(int[] layerSizes) {
        super(layerSizes);
    }

    public SigmoidCrossEntropyNetwork(NetworkVector vector) {
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
        DoubleMatrix ones = DoubleMatrix.ones(m.rows, m.columns);
        double epsilon = 1e-10; // Small epsilon value to avoid logarithm of zero

        DoubleMatrix logM = MatrixFunctions.log(m.add(epsilon));
        DoubleMatrix logOneMinusM = MatrixFunctions.log(ones.sub(m).add(epsilon));

        return (y.mul(logM).add((ones.sub(y)).mul(logOneMinusM))).neg();
    }

    @Override
    public DoubleMatrix costPrime(DoubleMatrix m, DoubleMatrix y) {
        return m.sub(y);
    }
}
