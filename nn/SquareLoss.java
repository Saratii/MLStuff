package nn;

import java.util.ArrayList;
import java.util.List;

public class SquareLoss {
    static List<Double> calculate(List<Matrix> inputs, List<Integer> actual){
        List<Double> losses = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++){
            Double loss = 0.0;
            for(int j = 0; j < inputs.get(i).rows; j++){
                if(j == actual.get(i)){
                    loss += Math.pow((1 - inputs.get(i).get(j, 0)), 2);
                } else {
                    loss += Math.pow((0 - inputs.get(i).get(j, 0)), 2);
                }
            }
            losses.add(loss / 2.0);
        }
        return losses;
    }
    static List<Matrix> backward(List<Matrix> inputs, List<Integer> actual){
        List<Matrix> values = new ArrayList<>();
        for(int i = 0; i<inputs.size(); i++){
            Matrix value = new Matrix(1, inputs.get(i).cols);
            for(int j = 0; j < value.cols; j++){
                if(j == actual.get(i)){
                    value.set(0, j, 1 - inputs.get(i).get(0, j));
                } else {
                    value.set(0, j, 0 - inputs.get(i).get(0, j));
                }
            }
            values.add(value);
        }
        return values;
    }
}
