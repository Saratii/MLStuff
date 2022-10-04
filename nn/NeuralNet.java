package nn;
import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    int numInputs;
    int numClasses;
    List<Layer> layers;
    public NeuralNet(int numInputs, int numClasses, List<Integer> numNodesInHiddenLayers){
        this.numInputs = numInputs;
        this.numClasses = numClasses;
        layers = new ArrayList<>();
        for(int i = 0; i<numNodesInHiddenLayers.size(); i++){
            if(i == 0){
                layers.add(new ReLULayer(numInputs, numNodesInHiddenLayers.get(i)));
            } else {
                layers.add(new ReLULayer(numNodesInHiddenLayers.get(i-1), numNodesInHiddenLayers.get(i)));
            }
        }
        layers.add(new SoftmaxLayer(numNodesInHiddenLayers.get(numNodesInHiddenLayers.size() - 1), numClasses));
    }
    public void train(List<List<Double>> data, List<Integer> actual) throws Exception{
        List<Matrix> values = new ArrayList<>();
        for(List<Double> point: data){
            Matrix value = new Matrix(1, point.size());
            for(int i = 0; i < point.size(); i++){
                value.values.get(0).set(i, point.get(i));
            }
            for(Layer layer: layers){
                value = layer.forward(value);
            }
            values.add(value);
        }
        List<Double> loss = SquareLoss.calculate(values, actual);
        System.out.println(loss);
    }
    public int classify(List<Double> input) throws Exception{
        if(input.size()!=numInputs){
            throw new Exception("bad input size");
        }
        Matrix layerOutput = new Matrix(input.size(), 1);
        for(int i = 0; i < input.size(); i++){
            layerOutput.values.get(i).set(0, input.get(i));
        }
        for(Layer layer: layers){
            layerOutput = layer.forward(layerOutput);
        }
        double best = 0;
        int bestIndex = 0;
        for(int i = 0; i < numClasses; i++){
            if(layerOutput.values.get(i).get(0) > best){
                best = layerOutput.values.get(i).get(0);
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
