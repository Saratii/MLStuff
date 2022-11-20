package nn;

import java.util.Arrays;
import java.util.List;

public class Main {
    static int NUM_CLASSES = 3;
    static int NUM_NODES = 10;
    static int LOG_FREQUENCY = 10;
    public static void main(String[] args) throws Exception {

        DataProcessor dp = new DataProcessor(); 
        dp.processTrainingData();
        dp.processTestingData();
        NeuralNet nn = new NeuralNet(dp.trainingData.get(0).size(), NUM_CLASSES, Arrays.asList(NUM_NODES));
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
            if(dp.testingActual.get(i).get(0) == 0.0) {
                System.out.println("Actual: lizard");
            } else if(dp.testingActual.get(i).get(0) == 1.0){
                System.out.println("Actual: lizzo");
            }
        }
        
        //lizzo is 1,0,0
        //lizzard is 0,1,0
        //cracker bear is 0,0,1
    }
    public static double average(List<Double> list) {
        double average = 0.0; 
        for (double i : list)
        average += i;
        return average/list.size();
    }
}