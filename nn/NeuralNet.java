package nn;

import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    int numInputs;
    int numClasses;
    List<Layer> layers;
    LossFunction lossFunction;
    static double ALPHA = 1000;

    // public NeuralNet(int numInputs, int numClasses, List<Integer> numNodesInHiddenLayers) {
    //     this.numInputs = numInputs;
    //     this.numClasses = numClasses;
    //     layers = new ArrayList<>();
    //     for (int i = 0; i < numNodesInHiddenLayers.size(); i++) {
    //         if (i == 0) {
    //             layers.add(new LogLayer(numInputs, numNodesInHiddenLayers.get(i)));
    //         } else {
    //             layers.add(new LogLayer(numNodesInHiddenLayers.get(i - 1), numNodesInHiddenLayers.get(i)));
    //         }
    //     }
    //     layers.add(new LogLayer(numNodesInHiddenLayers.get(numNodesInHiddenLayers.size() - 1), numClasses));
    //     lossFunction = new SquareLoss();
    // }

    public NeuralNet(int numInputs, int numClasses, List<Layer> layers) {
        this.numInputs = numInputs;
        this.numClasses = numClasses;
        this.layers = layers;
        lossFunction = new SquareLoss();
    }

    public List<Double> train(List<List<Double>> data, List<List<Double>> actual, int iteration)
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
        
        List<Matrix> derivatives = lossFunction.backward(values, actual);
        for (int i = layers.size() - 1; i >= 0; i--) {
            derivatives = layers.get(i).backward(derivatives);
        }
        return loss;
    }

    public void classify(List<List<Double>> input) throws Exception {
        List<Matrix> values = new ArrayList<>();
        for (List<Double> point : input) {
            Matrix value = new Matrix(1, point.size());
            for (int i = 0; i < point.size(); i++) {
                value.set(0, i, point.get(i));
            }
            values.add(value);
        }
        for (Layer layer : layers) {
            values = layer.forward(values);
        }
        
        System.out.println('\n');
        System.out.println(values);        
        if(values.get(0).maxPosition() == 0){
            System.out.println("Predicted: Elephant");
        } else if(values.get(0).maxPosition() == 1){
            System.out.println("Predicted: Squirrel");
        // } else if(values.get(0).maxPosition() == 2){
        //     System.out.println("Predicted: Cracker Bear");
        }
    }
}
