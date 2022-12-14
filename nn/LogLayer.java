package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LogLayer implements Layer {

    int numInputs;
    int numOutputs;
    Matrix weights;
    Matrix biases;
    List<Matrix> inputs;
    List<Matrix> outputs;

    public LogLayer(int numInputs, int numOutputs) {
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

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        inputs = values;
        outputs = new ArrayList<>();
        for (Matrix value : values) {
            outputs.add(log(value.multiply(weights).add(biases)));
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> outputs = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            Matrix value = values.get(i);
            Matrix output = new Matrix(1, value.cols);
            for (int j = 0; j < numOutputs; j++) {
                output.set(0, j, value.get(0, j) * this.outputs.get(i).get(0, j) * (1 - this.outputs.get(i).get(0, j)));
            }
            weights = weights.subtract(inputs.get(i).T().multiply(output).divide(NeuralNet.ALPHA));
            biases = biases.subtract(output.divide(NeuralNet.ALPHA));
            output = output.multiply(weights.T());
            outputs.add(output);
        }
        return outputs;
    }

    private Matrix log(Matrix value) {
        Matrix output = new Matrix(value.rows, value.cols);
        for (int i = 0; i < value.rows; i++) {
            for (int j = 0; j < value.cols; j++) {
                output.set(i, j, 1.0 / (1.0 + Math.pow(Math.E, -1 * value.get(i, j))));
            }
        }
        return output;
    }
}