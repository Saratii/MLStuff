package nn;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
// import java.util.Random;

// import javax.swing.ImageIcon;
// import javax.swing.JFrame;
// import javax.swing.JLabel;

public class Main {
    
    public static void main(String[] args) throws Exception {
        File lizzardTestFolderPath = new File("/Users/propleschmaren/Desktop/MLStuff/TestLizards");
        File lizzoTestFolderPath = new File("/Users/propleschmaren/Desktop/MLStuff/TestLizzos");
        File[] lizardTestFiles = lizzardTestFolderPath.listFiles();
        File[] lizzoTestFiles = lizzoTestFolderPath.listFiles();
        // var frame = new JFrame();
        // String imgPath = "";
        List<File> testFiles = new ArrayList<>();
        for(int i = 0; i < lizardTestFiles.length; i++){
            testFiles.add(lizardTestFiles[i]);
        }
        for(int i = 0; i < lizzoTestFiles.length; i++){
            testFiles.add(lizzoTestFiles[i]);
        }

        DataProcessor dp = new DataProcessor();
        dp.processTrainingData();
        dp.processTestingData();
        NeuralNet nn = new NeuralNet(dp.trainingData.get(0).size(), 2, Arrays.asList(2));
        for (int i = 0; i < 4000; i++) {
            nn.train(dp.trainingData, dp.trainingActual, i + 1, 10);
        }

        for(int i = 0; i < dp.testingActual.size(); i++){
            nn.classify(Arrays.asList(dp.testingData.get(i))); 
            if(dp.testingActual.get(i).get(0) == 0.0) {
                System.out.println("Actual: lizard");
            } else if(dp.testingActual.get(i).get(0) == 1.0){
                System.out.println("Actual: lizzo");
            }
            // imgPath = testFiles.get(i).toString();
        }
        //lizzo is 1,0
        //lizzard is 0,1
        
        
        

        // var icon = new ImageIcon(imgPath);
        // var Jlabel = new JLabel(icon);
        // frame.add(Jlabel);
        // frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // frame.pack();
        // frame.setVisible(true);

    }
}
