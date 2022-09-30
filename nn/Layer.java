package nn;

import java.util.Random;

public abstract class Layer {
    int numInputs;
    int numOutputs;
    Matrix weights;
    Matrix biases;
    Matrix outputs;

    public Layer(int numInputs, int numOutputs){
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        weights = new Matrix(numInputs, numOutputs);
        biases = new Matrix(numOutputs, 1);
        Random r = new Random();
        for(int i = 0; i<weights.values.size(); i++){
            biases.values.get(i).set(i, 2 * r.nextDouble() - 1);
            for(int j = 0; j<weights.values.get(i).size(); j++){
                weights.values.get(i).set(j, 2 * r.nextDouble() - 1);
            }
        }
    }
    public abstract Matrix forward(Matrix values) throws Exception;
    public abstract Matrix backward(Matrix values);
}
