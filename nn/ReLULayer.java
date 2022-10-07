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
        for (Matrix value : values) {
            outputs.add(ReLU(value.multiply(weights.add(biases))));
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) {
        List<Matrix> jacobians = new ArrayList<>();
        for (Matrix value : values) {
            Matrix jacobian = new Matrix(value.rows, value.cols);
            for (int i = 0; i < value.rows; i++) {
                for (int j = 0; j < value.cols; j++) {
                    if (value.values.get(i).get(j) >= 0) {
                        jacobian.values.get(i).set(j, 1.0);
                    } else {
                        jacobian.values.get(i).set(j, 0.0);
                    }
                }
            }
            jacobians.add(jacobian);
        }
        return jacobians;
    }

    public Matrix ReLU(Matrix values) {
        assert (values.cols == 1);
        for (int i = 0; i < values.rows; i++) {
            if (values.values.get(i).get(0) < 0) {
                values.values.get(i).set(0, 0.0);
            }
        }
        return values;
    }
}