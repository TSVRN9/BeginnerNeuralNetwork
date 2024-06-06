package me.tsvrn9.beginnerneuralnetwork.neuralnetwork;

import me.tsvrn9.beginnerneuralnetwork.Config;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.dataset.MnistDataset;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.dataset.MnistReader;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.functions.Functions;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;

import java.io.IOException;
import java.util.Random;

class NeuralNetworkTest {
    @org.junit.jupiter.api.Test
    void backpropagation() throws IOException {
        MnistDataset dataset = MnistReader.readData(Config.TESTING_IMAGES, Config.TESTING_LABELS);
        assert dataset != null;
        NetworkVector vector = new NetworkVector(new int[] {
                dataset.columns * dataset.rows,
                20,
                10,
        });
        NeuralNetwork network = new NeuralNetwork(vector, Functions.sigmoid, Functions.squared);

        Random random = new Random();
        double epsilon = 0.1;

        for (int i = 0; i < 5; i++) {
            int randomIndex = random.nextInt(dataset.length());
            DoubleMatrix input = dataset.getData(randomIndex);
            DoubleMatrix label = dataset.getLabelMatrix(randomIndex);

            NetworkVector gradient = network.backpropagation(input, label);
            NetworkVector checker = backpropagationChecker(network, input, label);

            NetworkVector difference = gradient.sub(checker);

            difference.foreach(d -> Assertions.assertTrue(Math.abs(d) < epsilon, "Yikes, the gradient don't work!"));
        }
    }

    private NetworkVector backpropagationChecker(NeuralNetwork network, DoubleMatrix input, DoubleMatrix label) {
        double epsilon = 1e-4; // Small perturbation for gradient approximation

        // Initialize a new NetworkVector to store the gradient approximation
        NetworkVector gradientApproximation = new NetworkVector(network.vector.layerSizes);

        // Iterate over each weight and bias in the network
        for (int layer = 1; layer < network.vector.numLayers; layer++) {
            int currentLayerSize = network.vector.layerSizes[layer];
            int previousLayerSize = network.vector.layerSizes[layer - 1];

            // Iterate over each weight in the current layer
            for (int i = 0; i < currentLayerSize; i++) {
                for (int j = 0; j < previousLayerSize; j++) {
                    // Approximate the gradient for the weight (i, j)
                    double originalWeight = network.vector.weights[layer].get(i, j);

                    // Perturb the weight by adding epsilon
                    network.vector.weights[layer].put(i, j, originalWeight + epsilon);
                    DoubleMatrix outputPlus = network.input(input);
                    double costPlus = network.cost.f(outputPlus, label).sum();

                    // Perturb the weight by subtracting epsilon
                    network.vector.weights[layer].put(i, j, originalWeight - epsilon);
                    DoubleMatrix outputMinus = network.input(input);
                    double costMinus = network.cost.f(outputMinus, label).sum();

                    // Calculate the gradient approximation using central difference formula
                    double gradient = (costPlus - costMinus) / (2 * epsilon);
                    gradientApproximation.weights[layer].put(i, j, gradient);

                    // Reset the weight to its original value
                    network.vector.weights[layer].put(i, j, originalWeight);
                }
            }

            // Iterate over each bias in the current layer
            for (int i = 0; i < currentLayerSize; i++) {
                // Approximate the gradient for the bias i
                double originalBias = network.vector.biases[layer].get(i);

                // Perturb the bias by adding epsilon
                network.vector.biases[layer].put(i, originalBias + epsilon);
                DoubleMatrix outputPlus = network.input(input);
                double costPlus = network.cost.f(outputPlus, label).sum();

                // Perturb the bias by subtracting epsilon
                network.vector.biases[layer].put(i, originalBias - epsilon);
                DoubleMatrix outputMinus = network.input(input);
                double costMinus = network.cost.f(outputMinus, label).sum();

                // Calculate the gradient approximation using central difference formula
                double gradient = (costPlus - costMinus) / (2 * epsilon);
                gradientApproximation.biases[layer].put(i, gradient);

                // Reset the bias to its original value
                network.vector.biases[layer].put(i, originalBias);
            }
        }

        return gradientApproximation;
    }
}