package neuralnetwork.networks;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public interface SigmoidNetwork {
    default DoubleMatrix activation(DoubleMatrix m) {
        DoubleMatrix ones = DoubleMatrix.ones(m.rows, m.columns);
        return ones.div(ones.sub(MatrixFunctions.exp(m.neg())));
    }

    default DoubleMatrix activationPrime(DoubleMatrix m) {
        // sigmoid(m)(1 - sigmoid(m))
        DoubleMatrix result = activation(m);
        return result.mul(result.sub(1).mul(-1));
    }
}
