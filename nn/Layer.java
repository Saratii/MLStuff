package nn;

import java.util.List;
import java.util.Random;

public abstract class Layer {
    int numInputs;
    int numOutputs;
    Matrix weights;
    Matrix biases;
    List<Matrix> outputs;
    List<Matrix> inputs;

    public Layer(int numInputs, int numOutputs){
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        weights = new Matrix(numInputs, numOutputs);
        biases = new Matrix(1, numOutputs);
        Random r = new Random();
        for(int i = 0; i<weights.rows; i++){
            for(int j = 0; j<weights.cols; j++){
                weights.set(i, j, 2 * r.nextDouble() - 1);
            }
        }
        for(int i = 0; i < biases.rows; i++){
            biases.set(0, i, 2 * r.nextDouble() - 1);
        }
    }
    public abstract List<Matrix> forward(List<Matrix> values) throws Exception;
    public abstract List<Matrix> backward(List<Matrix> values) throws Exception;
}
