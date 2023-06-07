package neuralnetwork;

import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;

public class NetworkVector {
    public final DoubleMatrix[] weights;
    // column vector
    public final DoubleMatrix[] biases;
    public final int[] layerSizes;
    public final int numLayers;

    public NetworkVector(int[] layerSizes) {
        int numLayers = layerSizes.length;

        // first value will be null... (it's for the readability)
        DoubleMatrix[] weights = new DoubleMatrix[numLayers];
        DoubleMatrix[] biases = new DoubleMatrix[numLayers];

        for (int l = 1; l < numLayers; l++) {
            int currentLayerSize = layerSizes[l];
            int previousLayerSize = layerSizes[l - 1];

            DoubleMatrix w = DoubleMatrix.rand(currentLayerSize, previousLayerSize).sub(0.5);
            DoubleMatrix b = DoubleMatrix.rand(currentLayerSize).sub(0.5);

            weights[l] = w;
            biases[l] = b;
        }

        this.weights = weights;
        this.biases = biases;
        this.layerSizes = layerSizes;
        this.numLayers = numLayers;
    }

    public NetworkVector(double[][][] weights, double[][] biases, int[] layerSizes) {
        int numLayers = layerSizes.length;

        DoubleMatrix[] weightMatricies = new DoubleMatrix[numLayers];
        DoubleMatrix[] biasMatricies = new DoubleMatrix[numLayers];

        for (int i = 0; i < weights.length; i++) {
            double[][] w = weights[i];
            double[] b = biases[i];

            weightMatricies[i] = new DoubleMatrix(w);
            biasMatricies[i] = new DoubleMatrix(b);
        }

        this.weights = weightMatricies;
        this.biases = biasMatricies;
        this.layerSizes = layerSizes;
        this.numLayers = numLayers;
    }

    public NetworkVector(DoubleMatrix[] weights, DoubleMatrix[] biases, int[] layerSizes, int numLayers) {
        this.weights = weights;
        this.biases = biases;
        this.layerSizes = layerSizes;
        this.numLayers = numLayers;
    }

    public NetworkVector clone() {
        return new NetworkVector(weights, biases, layerSizes, numLayers);
    }

    public NetworkVector add(NetworkVector o) {
        NetworkVector v = clone();
        for (int i = 1; i < numLayers; i++) {
            v.weights[i] = v.weights[i].add(o.weights[i]);
            v.biases[i] = v.biases[i].add(o.biases[i]);
        }
        return v;
    }

    public NetworkVector div(double d) {
        NetworkVector v = clone();
        for (int i = 1; i < numLayers; i++) {
            v.weights[i] = v.weights[i].div(d);
            v.biases[i] = v.biases[i].div(d);
        }
        return v;
    }

    public NetworkVector mul(double d) {
        NetworkVector v = clone();
        for (int i = 1; i < numLayers; i++) {
            v.weights[i] = v.weights[i].mul(d);
            v.biases[i] = v.biases[i].mul(d);
        }
        return v;
    }

    public static NetworkVector zeros(NetworkVector from) {
        int numLayers = from.layerSizes.length;

        // first value will be null... (it's for the readability)
        DoubleMatrix[] weights = new DoubleMatrix[numLayers];
        DoubleMatrix[] biases = new DoubleMatrix[numLayers];

        for (int l = 1; l < numLayers; l++) {
            int currentLayerSize = from.layerSizes[l];
            int previousLayerSize = from.layerSizes[l - 1];

            DoubleMatrix w = DoubleMatrix.zeros(currentLayerSize, previousLayerSize);
            DoubleMatrix b = DoubleMatrix.zeros(currentLayerSize);

            weights[l] = w;
            biases[l] = b;
        }
        return new NetworkVector(weights, biases, from.layerSizes, from.numLayers);
    }

    @Override
    public String toString() {
        return "neuralnetwork.NeuralNetworkVector{" +
                "weights=" + Arrays.toString(weights) +
                ", biases=" + Arrays.toString(biases) +
                ", layerSizes=" + Arrays.toString(layerSizes) +
                ", numLayers=" + numLayers +
                '}';
    }

    public NetworkVector sub(NetworkVector o) {
        NetworkVector v = clone();
        for (int i = 1; i < numLayers; i++) {
            v.weights[i] = v.weights[i].sub(o.weights[i]);
            v.biases[i] = v.biases[i].sub(o.biases[i]);
        }
        return v;
    }

    public void foreach(DoubleConsumer f) {
        for (int l = 1; l < numLayers; l++) {
            DoubleMatrix weights = this.weights[l];
            DoubleMatrix biases = this.biases[l];

            for (int i = 0; i < weights.length; i++) {
                f.accept(weights.get(i));
            }
            for (int i = 0; i < biases.length; i++) {
                f.accept(biases.get(i));
            }
        }
    }
}
