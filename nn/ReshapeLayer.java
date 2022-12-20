package nn;

import java.util.ArrayList;
import java.util.List;

public class ReshapeLayer implements Layer {

    int inputRows;
    int inputCols;
    int outputRows;
    int outputCols;

    public ReshapeLayer(int inputRows, int inputCols, int outputRows, int outputCols) throws Exception {
        if(inputRows * inputCols != outputRows * outputCols) {
            throw new Exception("Mismatched Dimensions");
        }
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.outputRows = outputRows;
        this.outputCols = outputCols;
    }

    @Override
    public List<Matrix> forward(List<Matrix> values) throws Exception {
        /**
         * Matrixes should be more like how numpy handles multi-dimensional matrixes, 
         * where all data is stored in a single one dimensional region,
         * and the shape is just a computation to translate the (i,j) coordinates into a single coordinate,
         * but I don't want to rewrite the matrixes right now,
         * so we're literally recreating each matrix and copying the data into the new shape.
         * This is wildly inefficient.
         */
        List<Matrix> outputs = new ArrayList<>();
        for(Matrix value: values) {
            Matrix output = new Matrix(outputRows, outputCols);
            for(int i = 0; i < value.rows; i++) {
                for(int j = 0; j < value.cols; j++) {
                    int index = i * value.cols + j;
                    output.set(index / output.cols, index % output.cols, value.get(i, j));
                }
            }
            outputs.add(output);
        }
        return outputs;
    }

    @Override
    public List<Matrix> backward(List<Matrix> values) throws Exception {
        List<Matrix> outputs = new ArrayList<>();
        for(Matrix value: values) {
            Matrix output = new Matrix(inputRows, inputCols);
            for(int i = 0; i < value.rows; i++) {
                for(int j = 0; j < value.cols; j++) {
                    int index = i * value.cols + j;
                    output.set(index / output.cols, index % output.cols, value.get(i, j));
                }
            }
            outputs.add(output);
        }
        return outputs;
    }

}
