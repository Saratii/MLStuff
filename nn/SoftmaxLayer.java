package nn;

public class SoftmaxLayer extends Layer{

    public SoftmaxLayer(int numInputs, int numOutputs){
        super(numInputs, numOutputs);
    }

    @Override
    public Matrix forward(Matrix values) throws Exception{
        outputs = softMax(values.multiply(weights).add(biases));
        return outputs;
    }

    @Override
    public Matrix backward(Matrix values) {
        assert(values.cols == 1);
        Matrix jacobian = new Matrix(values.rows, values.rows);
        for(int i = 0; i < values.rows; i++){
            for(int j = 0; j < values.rows; j++){
                if(i==j){
                    jacobian.values.get(i).set(j, outputs.values.get(i).get(0) * (1 - outputs.values.get(j).get(0)));
                } else {
                    jacobian.values.get(i).set(j, outputs.values.get(i).get(0) * (0 - outputs.values.get(j).get(0)));
                }
            }
        }
        return jacobian;
    }
    public Matrix softMax(Matrix values){
        assert(values.cols == 1);
        Double ex = 0.0;
        for(int i = 0; i < values.rows; i++){
            ex += Math.pow(Math.E, values.values.get(i).get(0));
        }
        Matrix newValues = new Matrix(values.rows, values.cols);
        for(int i = 0; i < values.rows; i++){
            newValues.values.get(i).set(0, Math.pow(Math.E, values.values.get(i).get(0) / ex));
        }
        return newValues;
    }
}
