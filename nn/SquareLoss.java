package nn;

import java.util.ArrayList;
import java.util.List;

public class SquareLoss implements LossFunction{
    public List<Double> calculate(List<Matrix> inputs, List<List<Double>> actual){
        List<Double> losses = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++){
            Double loss = 0.0;
            for(int j = 0; j < inputs.get(i).cols; j++){
                loss += Math.pow((actual.get(i).get(j) - inputs.get(i).get(0, j)), 2);  
            }
            losses.add(loss / 2.0);
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
