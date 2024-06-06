package me.tsvrn9.beginnerneuralnetwork;

import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.NetworkVector;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.NeuralNetwork;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.dataset.Dataset;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.dataset.MnistReader;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.functions.Functions;
import org.jblas.DoubleMatrix;
import me.tsvrn9.beginnerneuralnetwork.serialization.SimpleSerialization;

import java.io.IOException;

public class Training {
    public static void main(String[] args) throws IOException {
        Dataset<Integer> dataset = MnistReader.readData(Config.TRAINING_IMAGES, Config.TRAINING_LABELS);
        // Dataset<Boolean> dataset = new XorDataset();
        assert dataset != null;
        NetworkVector vector = SimpleSerialization.load(Config.FILE_PATH, NetworkVector.class);
        // NetworkVector vector = new NetworkVector(new int[] { dataset.numInputs(), 500, 500, 80, dataset.numOutputs() });
        NeuralNetwork network = new NeuralNetwork(vector, Functions.sigmoid, Functions.crossEntropy);

        for (int i = 0; i < Config.ITERATIONS_HUNDREDS; i++) {
            System.out.println();
            System.out.println(i + "/" + Config.ITERATIONS_HUNDREDS);

            int randI = (int) (Math.random() * dataset.length());
            DoubleMatrix output = network.input(dataset.getData(randI));
            System.out.println("Cost: " + network.cost.f(output, dataset.getLabelMatrix(randI)).sum());
            System.out.println("Output Matrix: " + output);
            System.out.println("What it thinks: " + dataset.getPredictedValue(output) + " with " + output.get(output.argmax()) * 100 + "%");
            System.out.println("What it is: " + dataset.getLabel(randI) + " guessed at " + dataset.getConfidence(output, dataset.getLabel(randI)) * 100 + "%");
            System.out.flush();

            network.train(dataset, 100, Config.BATCH_SIZE, Config.LEARNING_RATE);
        }

        cleanup(network);
    }

    private static void cleanup(NeuralNetwork network) {
        System.out.println("Saving...");
        System.out.flush();
        SimpleSerialization.save(network.vector, Config.FILE_PATH);
        System.out.println("Saved");
        System.out.flush();
        System.exit(0);
    }
}
