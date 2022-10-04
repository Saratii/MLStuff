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
}
