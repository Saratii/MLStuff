package nn;

import java.util.ArrayList;
import java.util.List;

public class CrossEntropyLoss implements LossFunction{
    public List<Double> calculate(List<Matrix> inputs, List<List<Double>> actual){
        List<Double> losses = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++){
            Double loss = 0.0;
            for(int j = 0; j<actual.get(i).size(); j++){
                loss -= Math.log(inputs.get(i).get(0, j)) * actual.get(i).get(j);
            }
            losses.add(loss);
        } 
        return losses;
    }
    public List<Matrix> backward(List<Matrix> inputs, List<List<Double>> actual){
        List<Matrix> values = new ArrayList<>();
        for(int i = 0; i<inputs.size(); i++){
            Matrix value = new Matrix(1, inputs.get(i).cols);
            for(int j = 0; j < value.cols; j++){
                value.set(0, j, inputs.get(i).get(0, j) - actual.get(i).get(j));

            }
            values.add(value);
        }
        return values;
    }
}
