package serialization;

public class Config {
    public static final String
            TRAINING_IMAGES = "emnist-digits-train-images-idx3-ubyte.gz",
            TRAINING_LABELS = "emnist-digits-train-labels-idx1-ubyte.gz",
            TESTING_IMAGES = "emnist-digits-test-images-idx3-ubyte.gz",
            TESTING_LABELS = "emnist-digits-test-labels-idx1-ubyte.gz";
    public static final int ITERATIONS_THOUSANDS = 3;
    public static final int BATCH_SIZE = 100;
    public static final double LEARNING_RATE = 1.1;
    public static final String FILE_PATH = "H:\\models\\mnist.json";
}
