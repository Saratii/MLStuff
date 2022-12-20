package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer implements Layer {

    int inputRows;
    int inputCols;
    List<Matrix> inputs;
    List<Matrix> outputs;
    List<Matrix> kernals;
    List<Matrix> biases;
    public ConvolutionLayer(int inputRows, int inputCols, int kernalSize, int kernalCount) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        Random r = new Random();
        kernals = new ArrayList<>();
        biases = new ArrayList<>();
        for(int i = 0; i < kernalCount; i++) {
            kernals.add(new Matrix(kernalSize, kernalSize));
            biases.add(new Matrix(inputRows - kernalSize + 1, inputCols - kernalSize + 1));
            for(int j = 0; j < kernalSize; j++) {
                for(int k = 0; k < kernalSize; k++) {
                    kernals.get(i).set(j,k,r.nextDouble() * 2 - 1);
                    biases.get(i).set(j,k,r.nextDouble() * 2 - 1);
                }
            }
        }

    }

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        inputs = values;
        outputs = new ArrayList<>();
        for (int i = 0; i < kernals.size(); i++) {
            for (Matrix value : values) {
                outputs.add(value.crossCorrelation(kernals.get(i), false).add(biases.get(i)));
            }
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> outputs = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            Matrix value = values.get(i);
            for (int j = 0; j < kernals.size(); j++) {
                Matrix kernalGradient = inputs.get(i).crossCorrelation(value, false);
                Matrix inputGradient = value.convolution(kernals.get(i), true);
                kernals.set(j, kernals.get(j).subtract(kernalGradient.divide(NeuralNet.ALPHA)));
                biases.set(j, biases.get(j).subtract(values.get(i).divide(NeuralNet.ALPHA)));
                outputs.add(inputGradient);
            }
        }
        return outputs;
    }
}