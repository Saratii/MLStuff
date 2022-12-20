package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SoftmaxLayer implements Layer {

    int numInputs;
    int numOutputs;
    Matrix weights;
    Matrix biases;
    List<Matrix> inputs;
    List<Matrix> outputs;

    public SoftmaxLayer(int numInputs, int numOutputs) {
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
            outputs.add(softMax(value.multiply(weights).add(biases)));
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> whateverIWantToCallIt = new ArrayList<>();
        for(int i = 0; i < values.size(); i++){
            Matrix value = values.get(i);
            whateverIWantToCallIt.add(value.multiply(weights));
        }
        for(int i = 0; i < values.size(); i++){
            Matrix value = values.get(i);
            weights = weights.subtract(value.T().multiply(inputs.get(i)).divide(NeuralNet.ALPHA));
            biases = biases.subtract(value).divide(NeuralNet.ALPHA);
        }
        return whateverIWantToCallIt;
    }

    public Matrix softMax(Matrix values) {
        Double ex = 0.0;
        for (int i = 0; i < values.cols; i++) {
            ex += Math.pow(Math.E, values.get(0, i));
        }
        Matrix newValues = new Matrix(values.rows, values.cols);
        for (int i = 0; i < values.cols; i++) {
            newValues.set(0, i, Math.pow(Math.E, values.get(0, i)) / ex);
        }
        return newValues;
    }
}
