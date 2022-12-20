package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DenseLayer implements Layer {

    Matrix weights;
    Matrix biases;
    List<Matrix> inputs;

    public DenseLayer(int numInputs, int numOutputs) {
        Random r = new Random();
        weights = new Matrix(numInputs, numOutputs);
        biases = new Matrix(numInputs, 1);
        for(int i = 0; i < numInputs; i++) {
            for(int j = 0; j < numOutputs; j++) {
                weights.set(i, j, 2 * r.nextDouble() - 1);
            }
            biases.set(i, 0, 2 * r.nextDouble() - 1);
        }
    }

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        inputs = values;
        List<Matrix> results = new ArrayList<>();
        for(Matrix value: values) {
            results.add(weights.multiply(value).add(biases));
        }
        return results;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> results = new ArrayList<>();
        for(int i = 0; i < values.size(); i++) {
            Matrix weightGradient = values.get(i).multiply(inputs.get(i).T());
            Matrix inputGradient = weights.T().multiply(values.get(i));
            weights = weights.subtract(weightGradient.multiply(NeuralNet.ALPHA));
            biases = biases.subtract(values.get(i).multiply(NeuralNet.ALPHA));
            results.add(inputGradient);
        }
        return results;
    }
    
}
