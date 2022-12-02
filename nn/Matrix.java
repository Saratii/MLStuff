package nn;
import java.util.ArrayList;
import java.util.List;

public class Matrix {
    private List<List<Double>> values;
    int rows;
    int cols;
    private boolean transposed = false;
    private Matrix transpose = null;
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
                newValues.set(i, j, get(i, j) + b.get(i, j));
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
                    sum += get(i,k) * b.get(k,j);
                }
                newValues.set(i, j, sum);
            }
        }
        return newValues;
    }
    public Matrix multiply(double b) {
        Matrix newValues = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newValues.set(i, j, get(i,j) * b);
            }
        }
        return newValues;
    }
    public double get(int row, int col) {
        if(transposed) {
            return transpose.get(col, row);
        }   
        return values.get(row).get(col);
    }
    public void set(int row, int col, double value) {
        if(transposed) {
            transpose.set(col, row, value);
        } else {
            values.get(row).set(col, value);
        }
    }
    public List<List<Double>> get() {
        if(transposed) {
            return transpose.values;
        }
        return values;
    }
    public void set(List<List<Double>> values) {
        if(transposed) {
            transpose.values = values;
        } else {
            this.values = values;
        }
    }
    public Matrix T() {
        if(transposed) {
            return transpose;
        }
        Matrix t = new Matrix(cols, rows);
        t.transpose = this;
        t.transposed = true;
        return t;
    }
    public String toString(){
        if(transposed){
            return transpose.values.toString();
        }
        return values.toString();
    }
    public Matrix subtract(Matrix b) throws Exception{
        if (rows != b.rows || cols != b.cols){
            throw new Exception("mismatched dims");
        }
        Matrix newValues = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                newValues.set(i, j, get(i, j) - b.get(i, j));
            }
        }
        return newValues;
    }
    public Matrix divide(double b) {
        Matrix newValues = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newValues.set(i, j, get(i,j) / b);
            }
        }
        return newValues;
    }
    public int maxPosition(){
        double max = 0;
        int maxPos = 0;
        for(int i = 0; i < cols; i++){
            if(get(0, i) > max){
                max = get(0, i);
                maxPos = i;
            }
        }
        return maxPos;
    }
}
