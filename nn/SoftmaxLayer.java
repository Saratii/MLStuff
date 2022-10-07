package nn;

import java.util.ArrayList;
import java.util.List;

public class SoftmaxLayer extends Layer {

    public SoftmaxLayer(int numInputs, int numOutputs) {
        super(numInputs, numOutputs);
    }

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        outputs = new ArrayList<>();
        for (Matrix value : values) {
            outputs.add(softMax(value.multiply(weights).add(biases)));
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) {
        List<Matrix> jacobians = new ArrayList<>();
        for (int k = 0; k < values.size(); k++){
            Matrix value = values.get(k);
            Matrix jacobian = new Matrix(value.rows, value.rows);
            for (int i = 0; i < value.rows; i++) {
                for (int j = 0; j < value.rows; j++) {
                    if (i == j) {
                        jacobian.set(i, j, outputs.get(k).get(i, 0) * (1 - outputs.get(k).get(j, 0)));
                    } else {
                        jacobian.set(i, j, outputs.get(k).get(i, 0) * (0 - outputs.get(k).get(j, 0)));
                    }
                }
            }
            jacobians.add(jacobian);
        }
        return jacobians;
    }

    public Matrix softMax(Matrix values) {
        assert (values.cols == 1);
        Double ex = 0.0;
        for (int i = 0; i < values.rows; i++) {
            ex += Math.pow(Math.E, values.get(i, 0));
        }
        Matrix newValues = new Matrix(values.rows, values.cols);
        for (int i = 0; i < values.rows; i++) {
            newValues.set(i, 0, Math.pow(Math.E, values.get(i, 0) / ex));
        }
        return newValues;
    }
}
