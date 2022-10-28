package nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Main {
    
    public static void main(String[] args) throws Exception {
        DataProcessor dp = new DataProcessor();
        dp.processTrainingData();
        dp.processTestingData();
        NeuralNet nn = new NeuralNet(dp.trainingData.get(0).size(), 2, Arrays.asList(2));
        for (int i = 0; i < 10000; i++) {
            nn.train(dp.trainingData, dp.trainingActual, i + 1, 1000);
        }
        nn.classify(Arrays.asList(dp.testingData.get(0)));        
        
        //dog is 1,0
        //cat is 0,1
        var frame = new JFrame();
        var icon = new ImageIcon("/Users/propleschmaren/Desktop/MLStuff/Cats/c1.jpeg");
        var label = new JLabel(icon);
        frame.add(label);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }
}