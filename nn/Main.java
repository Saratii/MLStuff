package nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    static int NUM_CLASSES = 2;
    static int NUM_NODES = 32;
    static int NUM_HIDDEN_LAYERS = 2;
    static int LOG_FREQUENCY = 1;
    public static void main(String[] args) throws Exception {

        DataProcessor dp = new DataProcessor(); 
        dp.processTrainingData();
        dp.processTestingData();
        List<Integer> layers = new ArrayList<>();
        for(int i = 0; i < NUM_HIDDEN_LAYERS; i++){
            layers.add(NUM_NODES);
        }
        NeuralNet nn = new NeuralNet(dp.trainingData.get(0).size(), NUM_CLASSES, layers);
        Long startTime = System.currentTimeMillis();
        for (int i = 0; i < 400; i++) {
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