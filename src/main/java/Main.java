import neuralnetwork.*;
import neuralnetwork.mnist.Dataset;
import neuralnetwork.mnist.MnistDataset;
import neuralnetwork.mnist.MnistReader;
import neuralnetwork.mnist.XorDataset;
import neuralnetwork.networks.SigmoidCrossEntropyNetwork;
import neuralnetwork.networks.SigmoidSquaredNetwork;
import org.jblas.DoubleMatrix;
import serialization.Config;
import serialization.SimpleSerialization;

import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static final String
            TRAINING_IMAGES = "emnist-digits-train-images-idx3-ubyte.gz",
            TRAINING_LABELS = "emnist-digits-train-labels-idx1-ubyte.gz",
            TESTING_IMAGES = "emnist-digits-test-images-idx3-ubyte.gz",
            TESTING_LABELS = "emnist-digits-test-labels-idx1-ubyte.gz";
    public static final int ITERATIONS_HUNDREDS = 20;
    private static final int BATCH_SIZE = 20;
    private static final double LEARNING_RATE = 1.0;

    public static void main(String[] args) throws IOException {
        Dataset<Integer> dataset = MnistReader.readData(TRAINING_IMAGES, TRAINING_LABELS);
        // Dataset<Boolean> dataset = new XorDataset();
        assert dataset != null;
        // NeuralNetwork network = new SigmoidCrossEntropyNetwork(new int[] { dataset.numInputs(), 500, 500, 80, dataset.numOutputs() });
        NeuralNetwork network = new SigmoidCrossEntropyNetwork(SimpleSerialization.load(Config.FILE_PATH, NetworkVector.class));

        for (int i = 0; i < ITERATIONS_HUNDREDS; i++) {
            System.out.println();
            System.out.println(i + "/" + ITERATIONS_HUNDREDS);
            int randI = (int) (Math.random() * dataset.length());
            DoubleMatrix output = network.input(dataset.getData(randI));
            System.out.println("Cost: " + network.cost(output, dataset.getLabelMatrix(randI)).sum());
            System.out.println("What it thinks: " + dataset.getPredictedValue(output) + " with " + output.get(output.argmax()) * 100 + "%");
            System.out.println("What it is: " + dataset.getLabel(randI) + " guessed at " + dataset.getConfidence(output, dataset.getLabel(randI)) * 100 + "%");
            System.out.flush();

            network.train(dataset, 100, BATCH_SIZE, LEARNING_RATE);
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
