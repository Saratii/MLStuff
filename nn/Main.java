package nn;

import java.util.Arrays;
// import java.util.Random;

// import javax.swing.ImageIcon;
// import javax.swing.JFrame;
// import javax.swing.JLabel;

public class Main {
    static int NUM_CLASSES = 3;
    public static void main(String[] args) throws Exception {

        DataProcessor dp = new DataProcessor();
        dp.processTrainingData();
        dp.processTestingData();
        NeuralNet nn = new NeuralNet(dp.trainingData.get(0).size(), NUM_CLASSES, Arrays.asList(5));
        for (int i = 0; i < 400; i++) {
            nn.train(dp.trainingData, dp.trainingActual, i + 1, 10);
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
}
