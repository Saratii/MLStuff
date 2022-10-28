package nn;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        NeuralNet nn = new NeuralNet(5, 2, Arrays.asList(2));
        List<List<Double>> data = Arrays.asList(
                Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0),
                Arrays.asList(5.0, 4.0, 3.0, 2.0, 1.0));
        List<List<Double>> actual = Arrays.asList(
                Arrays.asList(1.0, 0.0),
                Arrays.asList(0.0, 1.0));
        for (int i = 0; i < 100000; i++) {
            nn.train(data, actual, i + 1, 1000);
        }
        System.out.println(nn.classify(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)));
    }
}