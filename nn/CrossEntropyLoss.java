package nn;

import java.util.ArrayList;
import java.util.List;

public class CrossEntropyLoss {
    static List<Double> calculate(List<Matrix> inputs, List<Integer> actual){
        List<Double> losses = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++){
            Double loss = 0.0;
            loss -= Math.log(inputs.get(i).get(0, actual.get(i)));
            losses.add(loss);
        } 
        return losses;
    }
    static List<Matrix> backward(List<Matrix> inputs, List<Integer> actual){
        List<Matrix> values = new ArrayList<>();
        for(int i = 0; i<inputs.size(); i++){
            Matrix value = new Matrix(1, inputs.get(i).cols);
            for(int j = 0; j < value.cols; j++){
                if(actual.get(i) == j){
                    value.set(0, j, inputs.get(i).get(0, j) - 1);
                } else {
                    value.set(0, j, inputs.get(i).get(0, j));
                }
            }
            values.add(value);
        }
        return values;
    }
}
