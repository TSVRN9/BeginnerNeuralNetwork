package me.tsvrn9.beginnerneuralnetwork.neuralnetwork.functions;

import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class ActivationFunction {
    private final Function<DoubleMatrix, DoubleMatrix> f;
    private final Function<DoubleMatrix, DoubleMatrix> fPrime;

    public ActivationFunction(Function<DoubleMatrix, DoubleMatrix> f, Function<DoubleMatrix, DoubleMatrix> fPrime) {
        this.f = f;
        this.fPrime = fPrime;
    }

    public DoubleMatrix f(DoubleMatrix m) {
        return f.apply(m);
    }
    public DoubleMatrix fPrime(DoubleMatrix m) {
        return fPrime.apply(m);
    }
}
