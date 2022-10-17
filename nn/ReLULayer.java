package nn;

import java.util.ArrayList;
import java.util.List;

public class ReLULayer extends Layer {

    public ReLULayer(int numInputs, int numOutputs) {
        super(numInputs, numOutputs);
    }

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        outputs = new ArrayList<>();
        inputs = values;
        for (Matrix value : values) {
            outputs.add(ReLU(value.multiply(weights).add(biases)));
        }
        return outputs;
    }
    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> derivatives = new ArrayList<>();
        for(int i = 0; i < values.size(); i++){
            Matrix value = values.get(i);
            Matrix reluPrime = new Matrix(outputs.get(i).cols, outputs.get(i).cols);
            for(int j = 0; j < reluPrime.cols; j++){
                for(int k = 0; k < reluPrime.cols; k++){
                    if(outputs.get(i).get(0, k) > 0){
                        reluPrime.set(0, k, 1);
                    }
                }
            }
            derivatives.add(value.multiply(reluPrime).multiply(weights.T()));
        }
        for(int i = 0; i < values.size(); i++){
            Matrix reluPrime = new Matrix(outputs.get(i).cols, outputs.get(i).cols);
            for(int j = 0; j < reluPrime.cols; j++){
                for(int k = 0; k < reluPrime.cols; k++){
                    if(outputs.get(i).get(0, k) > 0){
                        reluPrime.set(0, k, 1);
                    }
                }
            }
            weights = weights.subtract(values.get(i).multiply(reluPrime).T().multiply(inputs.get(i)).divide(NeuralNet.ALPHA).T());
            biases = biases.subtract(values.get(i).multiply(reluPrime).divide(NeuralNet.ALPHA));
        }

        
        return derivatives;
    }

    public Matrix ReLU(Matrix values) {
        assert (values.rows == 1);
        for (int i = 0; i < values.cols; i++) {
            if (values.get(0, i) < 0) {
                values.set(0, i, 0.0);
            }
        }
        return values;
    }
}