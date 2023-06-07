package neuralnetwork;

import neuralnetwork.mnist.Dataset;
import org.jblas.DoubleMatrix;

import java.util.Random;

public abstract class NeuralNetwork {
    public NetworkVector vector;

    public NeuralNetwork(int[] layerSizes) {
        this.vector = new NetworkVector(layerSizes);
    }

    public NeuralNetwork(double[][][] weights, double[][] biases, int[] layerSizes) {
        this.vector = new NetworkVector(weights, biases, layerSizes);
    }

    public NeuralNetwork(NetworkVector vector) {
        this.vector = vector;
    }

    public DoubleMatrix input(DoubleMatrix inputs) {
        final int L = vector.numLayers - 1;

        DoubleMatrix a = inputs;

        for (int l = 1; l < L; l++) {
            DoubleMatrix w = vector.weights[l];
            DoubleMatrix b = vector.biases[l];
            DoubleMatrix z = w.mmul(a).add(b);
            a = activation(z);
        }
        // output activation function may differ
        {
            DoubleMatrix w = vector.weights[L];
            DoubleMatrix b = vector.biases[L];
            DoubleMatrix z = w.mmul(a).add(b);
            a = outputActivation(z);
        }

        return a;
    }

    public NeuralNetwork train(Dataset<?> dataset, int iterations, int batchSize, double learningRate) {
        Random rand = new Random();
        for (int i = 0; i < iterations; i++) {
            // average over
            NetworkVector gradient = NetworkVector.zeros(vector);
            for (int j = 0; j < batchSize; j++) {
                int ri = rand.nextInt(dataset.length());
                gradient = gradient.add(backpropagation(dataset.getData(ri), dataset.getLabelMatrix(ri)));
            }
            // gradient
            vector = vector.add(gradient.mul(-learningRate / batchSize));
        }
        return this;
    }

    public NetworkVector backpropagation(DoubleMatrix m, DoubleMatrix y) {
        final int L = vector.numLayers - 1;

        // feed forward
        DoubleMatrix[] activations = new DoubleMatrix[vector.numLayers];
        DoubleMatrix[] weightedInputs = new DoubleMatrix[vector.numLayers];

        activations[0] = m;

        for (int l = 1; l < L; l++) {
            DoubleMatrix a = activations[l - 1];

            // al = activation(w * al-1 + b)
            DoubleMatrix w = vector.weights[l];
            DoubleMatrix b = vector.biases[l];

            DoubleMatrix z = w.mmul(a).add(b);

            weightedInputs[l] = z;
            activations[l] = activation(z);
        }
        {
            // output layer activations
            DoubleMatrix a = activations[L - 1];

            DoubleMatrix w = vector.weights[L];
            DoubleMatrix b = vector.biases[L];

            DoubleMatrix z = w.mmul(a).add(b);

            weightedInputs[L] = z;
            activations[L] = outputActivation(z);
        }

        // backpropagation
        DoubleMatrix[] errors = new DoubleMatrix[vector.numLayers];

        // calculate error for the last layer, L
        errors[L] = costPrime(activations[L], y).mul(outputActivationPrime(weightedInputs[L]));

        // calculate error for every other layer
        for (int l = L - 1; l >= 1; l--) {
            errors[l] = vector.weights[l + 1].transpose().mmul(errors[l + 1]).mul(activationPrime(weightedInputs[l]));
        }

        // calculate updated weights and biases
        DoubleMatrix[] weightGradientMatrix = new DoubleMatrix[vector.numLayers];
        for (int l = 1; l < vector.numLayers; l++) {
            weightGradientMatrix[l] = errors[l].mmul(activations[l - 1].transpose());
        }

        return new NetworkVector(weightGradientMatrix, errors, vector.layerSizes, vector.numLayers);
    }

    public DoubleMatrix outputActivation(DoubleMatrix m) {
        return activation(m);
    }
    public DoubleMatrix outputActivationPrime(DoubleMatrix m) {
        return activationPrime(m);
    }
    public abstract DoubleMatrix activation(DoubleMatrix m);
    public abstract DoubleMatrix activationPrime(DoubleMatrix m);
    public abstract DoubleMatrix cost(DoubleMatrix m, DoubleMatrix y);
    public abstract DoubleMatrix costPrime(DoubleMatrix m, DoubleMatrix y);
}
