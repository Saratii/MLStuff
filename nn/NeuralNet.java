package nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNet {
    int numInputs;
    int numClasses;
    List<Layer> layers;
    LossFunction lossFunction;
    static double ALPHA = 50;

    public NeuralNet(int numInputs, int numClasses, List<Integer> numNodesInHiddenLayers) {
        this.numInputs = numInputs;
        this.numClasses = numClasses;
        layers = new ArrayList<>();
        for (int i = 0; i < numNodesInHiddenLayers.size(); i++) {
            if (i == 0) {
                layers.add(new LogLayer(numInputs, numNodesInHiddenLayers.get(i)));
            } else {
                layers.add(new LogLayer(numNodesInHiddenLayers.get(i - 1), numNodesInHiddenLayers.get(i)));
            }
        }
        layers.add(new LogLayer(numNodesInHiddenLayers.get(numNodesInHiddenLayers.size() - 1), numClasses));
        lossFunction = new SquareLoss();
    }

    public void train(List<List<Double>> data, List<List<Double>> actual, int iteration, int logFrequency)
            throws Exception {
        List<Matrix> values = new ArrayList<>();
        for (List<Double> point : data) {
            Matrix value = new Matrix(1, point.size());
            for (int i = 0; i < point.size(); i++) {
                value.set(0, i, point.get(i));
            }
            values.add(value);
        }
        for (Layer layer : layers) {
            values = layer.forward(values);
        }
        List<Double> loss = lossFunction.calculate(values, actual);
        if (iteration % logFrequency == 0) {
            System.out.println("Predicted: " + values);
            System.out.println("Loss: " + loss);
        }
        List<Matrix> derivatives = lossFunction.backward(values, actual);
        for (int i = layers.size() - 1; i >= 0; i--) {
            derivatives = layers.get(i).backward(derivatives);
        }
    }

    public int classify(List<Double> input) throws Exception {
        if (input.size() != numInputs) {
            throw new Exception("bad input size");
        }
        List<Matrix> layerOutput = Arrays.asList(new Matrix(input.size(), 1));
        for (int i = 0; i < input.size(); i++) {
            layerOutput.get(0).set(i, 0, input.get(i));
        }
        for (Layer layer : layers) {
            layerOutput = layer.forward(layerOutput);
        }
        double best = 0;
        int bestIndex = 0;
        for (int i = 0; i < numClasses; i++) {
            if (layerOutput.get(0).get(i, 0) > best) {
                best = layerOutput.get(0).get(i, 0);
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
