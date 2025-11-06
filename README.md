# BeginnerNeuralNetwork
## Motivation
This was my final project for AP CSA. I challenged myself to make this in Java instead of Python, which is traditionally used, so I would have to learn about perceptrons from the ground up. 
I more recently tidied this project up and refactored the various cost and activation functions to follow a functional paradigm instead of traditional inheritance.
But, the core of the project remains the same as when I wrote it in spring of 2023.

I ended up using JBlas as my linear algebra library, and gson to quickly serialize and deserialize objects. 
I also used ChatGPT to write much of the code within `DrawingApp`. I just don't really like Swing...
## Usage
Within the main package, `me.tsvrn9.beginnerneuralnetwork`, I have included two classes with main methods.
`Training` will train the MLP on the EMNIST dataset by default, included as resources. Values can be changed within `Config`. 
`DrawingApp` will allow for drawing numbers to test out the neural network.

You will have to change values within the config and training.

## Snippets I'm Proud Of
### Functions and Derivatives are represented by lambdas
```java
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
```
These lines define a new `CostFunction` object. 
`CostFunction` is basically a wrapper class associating a function and its derivative.
Before this, `NeuralNetwork` was an abstract class and any combination of cost functions and activation
functions had to be implemented in child classes.
### Initializing a `NetworkVector`
```java
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
        
        // rest omitted...
}
```
This is the constructor of `NetworkVector`. This actually took a surprising amount of time for me to conceptualize.
JBlas provides matricies, which I knew I wanted to use. So, I create a "3D" array of doubles by using an array of 2D DoubleMatricies.
I followed along with an online textbook, and I tried to maintain their conventions with how they index the weights and biases.
The first layer doesn't really "have" weights and definitely doesn't have biases associated with it.
So, I left it as null.

I defined a `NetworkVector` to basically represent all the weights and biases of a MLP.
In doing so, I'm able to just use gson to quickly serialize and deserialize this. 
It also allows for me to create a new object to represent how the weights and biases should change.
### Training a `NeuralNetwork`
```java
public NeuralNetwork train(Dataset<?> dataset, int iterations, int batchSize, double learningRate) {
    Random rand = new Random();
    for (int i = 0; i < iterations; i++) {
        // stochastic gradient descent
        NetworkVector gradient = NetworkVector.zeros(vector);
        for (int j = 0; j < batchSize; j++){
            int ri = rand.nextInt(dataset.length());
            gradient = gradient.add(backpropagation(dataset.getData(ri),dataset.getLabelMatrix(ri)));
        }
        vector = vector.add(gradient.mul(-learningRate / batchSize));
    }
    return this;
}
```
I like to think that my code here is easily understandable and clear. 
`NetworkVector.zeros` returns a `NetworkVector` with the same shape as the input vector.
This represents stochastic gradient descent for a given `batchsize` and repeats it `iterations` times.
This is primarily the reason I defined a `NetworkVector` instead 
of including weights and biases as fields directly within `NeuralNetwork`
