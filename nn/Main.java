package nn;

import java.util.Arrays;
import java.util.List;


public class Main{
    public static void main(String[] args) throws Exception{
        NeuralNet nn = new NeuralNet(2, 2, Arrays.asList(2));
        List<List<Double>> data = Arrays.asList(
            Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0),
            Arrays.asList(5.0, 4.0, 3.0, 2.0, 1.0)
        );
        List<List<Double>> actual = Arrays.asList(
            Arrays.asList(1.0, 0.0), Arrays.asList(0.0, 1.0)
        );
        for(int i = 0; i < 100; i++) {
            nn.train(data, actual);
        }
        // [1, 2, 3, 4, 5] = Forg //0
        // [5, 4, 3, 2, 1] = Pengoo //1
    }   
}