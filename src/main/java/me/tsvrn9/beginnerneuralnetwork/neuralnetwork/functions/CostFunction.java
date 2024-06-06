package me.tsvrn9.beginnerneuralnetwork.neuralnetwork.functions;

import org.jblas.DoubleMatrix;

import java.util.function.BiFunction;

public class CostFunction {
    private final BiFunction<DoubleMatrix, DoubleMatrix, DoubleMatrix> f;
    private final BiFunction<DoubleMatrix, DoubleMatrix, DoubleMatrix> fPrime;

    public CostFunction(BiFunction<DoubleMatrix, DoubleMatrix, DoubleMatrix> f, BiFunction<DoubleMatrix, DoubleMatrix, DoubleMatrix> fPrime) {
        this.f = f;
        this.fPrime = fPrime;
    }

    public DoubleMatrix f(DoubleMatrix m, DoubleMatrix y) {
        return f.apply(m, y);
    }
    public DoubleMatrix fPrime(DoubleMatrix m, DoubleMatrix y) {
        return fPrime.apply(m, y);
    }
}
