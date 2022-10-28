package nn;

import java.util.List;

public interface LossFunction {

    List<Double> calculate(List<Matrix> values, List<List<Double>> actual);

    List<Matrix> backward(List<Matrix> values, List<List<Double>> actual);

}
