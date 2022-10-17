package nn;

import java.util.ArrayList;
import java.util.List;

public class SoftmaxLayer extends Layer {

    public SoftmaxLayer(int numInputs, int numOutputs) {
        super(numInputs, numOutputs);
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
