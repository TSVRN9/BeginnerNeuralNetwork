package me.tsvrn9.beginnerneuralnetwork.neuralnetwork.functions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Functions {
    public static ActivationFunction sigmoid = new ActivationFunction(
            (m) -> {
                DoubleMatrix ones = DoubleMatrix.ones(m.rows, m.columns);
                return ones.div(ones.add(MatrixFunctions.exp(m.neg())));
            },
            (m) -> {
                // sigmoid(m)(1 - sigmoid(m))
                DoubleMatrix ones = DoubleMatrix.ones(m.rows, m.columns);
                DoubleMatrix result = ones.div(ones.add(MatrixFunctions.exp(m.neg())));
                return result.mul(result.sub(1).mul(-1));
            }
    );

    public static CostFunction squared = new CostFunction(
            (m, y) -> MatrixFunctions.pow(m.sub(y), 2),
            (m, y) -> m.sub(y).mul(2)
    );

    public static CostFunction crossEntropy = new CostFunction(
            (m, y) -> {
                DoubleMatrix ones = DoubleMatrix.ones(m.rows, m.columns);
                double epsilon = 1e-10; // Small epsilon value to avoid logarithm of zero

                DoubleMatrix logM = MatrixFunctions.log(m.add(epsilon));
                DoubleMatrix logOneMinusM = MatrixFunctions.log(ones.sub(m).add(epsilon));

                return (y.mul(logM).add((ones.sub(y)).mul(logOneMinusM))).neg();
            },
            DoubleMatrix::sub
    );
}
