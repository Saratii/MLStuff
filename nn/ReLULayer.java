package nn;

public class ReLULayer extends Layer{

    public ReLULayer(int numInputs, int numOutputs){
        super(numInputs, numOutputs);
    }

    @Override
    public Matrix forward(Matrix values) throws Exception{
        outputs = ReLU(values.multiply(weights).add(biases));
        return outputs;
    }

    @Override
    public Matrix backward(Matrix values) {
        Matrix jacobian = new Matrix(values.rows, values.cols);
        for(int i = 0; i < values.rows; i++){
            for(int j = 0; j < values.cols; j++){
                if(values.values.get(i).get(j) >= 0){
                    jacobian.values.get(i).set(j, 1.0);
                } else {
                    jacobian.values.get(i).set(j, 0.0);
                }
            }
        }
        return jacobian;
    }
    public Matrix ReLU(Matrix values){
        assert(values.cols == 1);
        for(int i = 0; i < values.rows; i++){
            if(values.values.get(i).get(0) < 0){
                values.values.get(i).set(0, 0.0);
            }
        }
        return values;
    }
}