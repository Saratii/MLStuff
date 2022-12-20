package nn;

import java.util.ArrayList;
import java.util.List;

public class SigmoidLayer implements Layer {

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        List<Matrix> outputs = new ArrayList<>();
        for (Matrix value: values) {
            outputs.add(sigmoid(value));
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> results = new ArrayList<>();
        for(Matrix value: values) {
            Matrix s = sigmoid(value);
            results.add(s.multiply(s.subtract(1.0).multiply(-1)));
        }
        return results;
    }

    private Matrix sigmoid(Matrix value) {
        Matrix output = new Matrix(value.rows, value.cols);
        for(int i = 0; i < value.rows; i++) {
            for(int j = 0; j < value.cols; j++) {
                output.set(i, j, 1 / (1 + Math.pow(Math.E, value.get(i, j))));
            }
        }
        return output;
    }
    
}
