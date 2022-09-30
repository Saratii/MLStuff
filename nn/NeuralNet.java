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
}
