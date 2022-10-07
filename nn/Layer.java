package nn;

import java.util.List;
import java.util.Random;

public abstract class Layer {
    int numInputs;
    int numOutputs;
    Matrix weights;
    Matrix biases;
    List<Matrix> outputs;

    public Layer(int numInputs, int numOutputs){
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        weights = new Matrix(numInputs, numOutputs);
        biases = new Matrix(1, numOutputs);
        Random r = new Random();
        for(int i = 0; i<weights.values.size(); i++){
            for(int j = 0; j<weights.values.get(i).size(); j++){
                weights.values.get(i).set(j, 2 * r.nextDouble() - 1);
            }
        }
        for(int i = 0; i < biases.values.size(); i++){
            biases.values.get(0).set(i, 2 * r.nextDouble() - 1);
        }
    }
    public abstract List<Matrix> forward(List<Matrix> values) throws Exception;
    public abstract List<Matrix> backward(List<Matrix> values);
}
