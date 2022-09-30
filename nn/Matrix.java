package nn;
import java.util.ArrayList;
import java.util.List;

public class Matrix {
    public List<List<Double>> values;
    int rows;
    int cols;
    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        values = new ArrayList<>();
        for(int i = 0; i < rows; i++){
            values.add(new ArrayList<>());
            for(int j = 0; j < cols; j++){
                values.get(i).add(0.0);
            }
        }
    }
    public Matrix add(Matrix b) throws Exception{
        if (rows != b.rows || cols != b.cols){
            throw new Exception("mismatched dims");
        }
        Matrix newValues = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                newValues.values.get(i).set(j, values.get(i).get(j) + b.values.get(i).get(j));
            }
        }
        return newValues;
    }
    public Matrix multiply(Matrix b) throws Exception{
        if(cols != b.rows){
            throw new Exception("mismatched dims");
        }
        Matrix newValues = new Matrix(rows, b.cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < b.cols; j++){
                double sum = 0;
                for(int k = 0; k < cols; k++){
                    sum += values.get(i).get(k) * b.values.get(k).get(j);
                }
                newValues.values.get(i).set(j, sum);
            }
        }
        return newValues;
    }
}
