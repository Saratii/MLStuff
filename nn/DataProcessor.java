package nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataProcessor {
    String st;
    List<List<Double>> trainingData = new ArrayList<>();
    List<List<Double>> testingData = new ArrayList<>();
    List<List<Double>> trainingActual = new ArrayList<>();
    List<List<Double>> testingActual = new ArrayList<>();

    public DataProcessor(){
    }
    public void processTrainingData() throws IOException{
        File xFile = new File("X.txt");
        File yFile = new File("Y.txt");
        BufferedReader br = new BufferedReader(new FileReader(xFile));
        BufferedReader br2 = new BufferedReader(new FileReader(yFile));
        List<String> lines = new ArrayList<>();
        List<String> linesY = new ArrayList<>();
        while ((st = br.readLine()) != null){lines.add(st);}
        while((st = br2.readLine()) != null){linesY.add(st);}
        br.close();
        br2.close();
        List<List<String>> lines2 = new ArrayList<>();
        for(int i = 0; i < lines.size(); i++){
            lines2.add(Arrays.asList(lines.get(i).split(",")));
        }
        for(int i = 0; i < lines2.size(); i++){
            trainingData.add(new ArrayList<>());
            for(int j = 0; j < lines2.get(0).size(); j++){
                trainingData.get(i).add(Double.parseDouble(lines2.get(i).get(j)));
            }
        }
        List<List<String>> linesY2 = new ArrayList<>();
        for(int i = 0; i < linesY.size(); i++){
            linesY2.add(Arrays.asList(linesY.get(i).split(",")));
        }
        for(int i = 0; i < linesY2.size(); i++){
            trainingActual.add(new ArrayList<>());
            for(int j = 0; j < linesY2.get(0).size(); j++){
                trainingActual.get(i).add(Double.parseDouble(linesY2.get(i).get(j)));
            }
        }
    }
    public void processTestingData() throws IOException{
        File txFile = new File("testX.txt");
        File tyFile = new File("testY.txt");
        BufferedReader br3 = new BufferedReader(new FileReader(txFile));
        BufferedReader br4 = new BufferedReader(new FileReader(tyFile));
        List<String> testLines = new ArrayList<>();
        List<String> testLinesY = new ArrayList<>();
        while((st = br3.readLine()) != null){testLines.add(st);}
        br3.close();
        while((st = br4.readLine()) != null){testLinesY.add(st);}
        br4.close();
        List<List<String>> testLines2 = new ArrayList<>();
        List<List<String>> testLinesY2 = new ArrayList<>();
        for(int i = 0; i < testLines.size(); i++){
            testLines2.add(Arrays.asList(testLines.get(i).split(",")));
        }
        for(int i = 0; i < testLinesY.size(); i++){
            testLinesY2.add(Arrays.asList(testLinesY.get(i).split(",")));
        }
        for(int i = 0; i < testLines2.size(); i++){
            testingData.add(new ArrayList<>());
            for(int j = 0; j < testLines2.get(0).size(); j++){
                testingData.get(i).add(Double.parseDouble(testLines2.get(i).get(j)));
            }
        }
        for(int i = 0; i < testLinesY2.size(); i++){
            testingActual.add(new ArrayList<>());
            for(int j = 0; j < testLinesY2.get(0).size(); j++){
                testingActual.get(i).add(Double.parseDouble(testLinesY2.get(i).get(j)));
            }
        }
    }
}
