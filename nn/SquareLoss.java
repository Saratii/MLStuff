package nn;

import java.util.ArrayList;
import java.util.List;

public class SquareLoss {
    static List<Double> calculate(List<Matrix> inputs, List<Integer> actual){
        List<Double> losses = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++){
            Double loss = 0.0;
            for(int j = 0; j < inputs.get(i).values.size(); j++){
                if(j == actual.get(i)){
                    loss += Math.pow((1 - inputs.get(i).values.get(j).get(0)), 2);
                } else {
                    loss += Math.pow((0 - inputs.get(i).values.get(j).get(0)), 2);
                }
            }
            losses.add(loss);
        }
        return losses;
    }
    static List<Matrix> backward(List<Matrix> inputs, List<Integer> actual){
        List<Matrix> values = new ArrayList<>();
        for(int i = 0; i<inputs.size(); i++){
            Matrix value = new Matrix(1, inputs.get(i).cols);
            for(int j = 0; j < value.cols; j++){
                if(j == actual.get(i)){
                    value.values.get(0).set(j, 2 * (1 - inputs.get(i).values.get(j).get(0)));
                } else {
                    value.values.get(0).set(j, 2 * (0 - inputs.get(i).values.get(j).get(0)));
                }
            }
            values.add(value);
        }
        return values;
    }
}
