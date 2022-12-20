package nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    static int NUM_CLASSES = 2;
    static int NUM_NODES = 32;
    static int NUM_HIDDEN_LAYERS = 2;
    static int LOG_FREQUENCY = 1;
    static int ITERATIONS = 500;
    public static void main(String[] args) throws Exception {

        DataProcessor dp = new DataProcessor(); 
        dp.processTrainingData();
        dp.processTestingData();
        List<Layer> layers = new ArrayList<>();
        layers.add(new ReshapeLayer(1,50 * 50, 50,50));
        layers.add(new ConvolutionLayer(50, 50, 4, 5));
        layers.add(new SigmoidLayer());
        layers.add(new ReshapeLayer(50,50,50*50,1));
        layers.add(new DenseLayer(5 * 50 * 50, 100));
        layers.add(new SigmoidLayer());
        layers.add(new DenseLayer(100,10));
        layers.add(new SigmoidLayer());
        NeuralNet nn = new NeuralNet(dp.trainingData.get(0).size(), NUM_CLASSES, layers);
        Long startTime = System.currentTimeMillis();
        for (int i = 0; i < ITERATIONS; i++) {
            List<Double> loss = nn.train(dp.trainingData, dp.trainingActual, i + 1);
            if (i % LOG_FREQUENCY == 0) {
                // System.out.println("\nPredicted: " + values);
                Long finishTime = System.currentTimeMillis();
                System.out.println("Iteration: "+i+" Average Loss: "+ average(loss) + " Run Time: " + (finishTime - startTime) / LOG_FREQUENCY + "ms");
                startTime = System.currentTimeMillis();
            }
        } 
       
        for(int i = 0; i < dp.testingActual.size(); i++){
            nn.classify(Arrays.asList(dp.testingData.get(i))); 
            if(dp.testingActual.get(i).get(1) == 1.0) {
                System.out.println("Actual: Squirrel");
            } else if(dp.testingActual.get(i).get(0) == 1.0){
                System.out.println("Actual: Elephant");
            // } else if(dp.testingActual.get(i).get(2) == 1.0){
            //     System.out.println("Actual: CrackerBear");
            }
        }
        //squirrel is 0,1
        //elephant is 1,0
    }
    public static double average(List<Double> list) {
        double average = 0.0; 
        for (double i : list){
            average += i;
        }
        return average/list.size();
    }
}