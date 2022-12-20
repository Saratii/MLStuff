package nn;
import java.util.ArrayList;
import java.util.Arrays;
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
    public Matrix subtract(double b) {
        Matrix newValues = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                newValues.set(i, j, get(i, j) - b);
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

    public Matrix crossCorrelation(Matrix b, boolean full) throws Exception {
        int resultRows = rows + (full ? 1 : -1) * (b.rows - 1);
        int resultCols = cols + (full ? 1 : -1) * (b.cols - 1);
        Matrix output = new Matrix(resultRows, resultCols);
        if(full) {
            for(int i_out = 0; i_out < output.rows; i_out++) {
                for(int j_out = 0; j_out < output.cols; j_out++) {
                    double sum = 0;
                    for(int i = 0; i < b.rows; i++) {
                        for(int j = 0; j < b.cols; j++) {
                            sum += (i - b.rows + i_out + 1 < 0 || j - b.cols + j_out + 1 < 0 || i - b.rows + i_out + 1 >= rows || j - b.cols + j_out + 1 >= cols
                                ? 0
                                : this.get(i - b.rows + i_out + 1, j - b.cols + j_out + 1)) * b.get(i, j);
                        }
                    }
                    output.set(i_out, j_out, sum);
                }
            }
        } else {
            for(int i_out = 0; i_out < output.rows; i_out++) {
                for(int j_out = 0; j_out < output.cols; j_out++) {
                    double sum = 0;
                    for(int i = 0; i < b.rows; i++) {
                        for(int j = 0; j < b.cols; j++) {
                            sum += this.get(i + i_out, j + j_out) * b.get(i, j);
                        }
                    }
                    output.set(i_out, j_out, sum);
                }
            }
        }

        return output;
    }

    public Matrix convolution(Matrix b, boolean full) throws Exception {
        int resultRows = rows + (full ? 1 : -1) * (b.rows - 1);
        int resultCols = cols + (full ? 1 : -1) * (b.cols - 1);
        Matrix output = new Matrix(resultRows, resultCols);
        if(full) {
            for(int i_out = 0; i_out < output.rows; i_out++) {
                for(int j_out = 0; j_out < output.cols; j_out++) {
                    double sum = 0;
                    for(int i = 0; i < b.rows; i++) {
                        for(int j = 0; j < b.cols; j++) {
                            sum += (i - b.rows + i_out + 1 < 0 || j - b.cols + j_out + 1 < 0 || i - b.rows + i_out + 1 >= rows || j - b.cols + j_out + 1 >= cols
                                ? 0
                                : this.get(i - b.rows + i_out + 1, j - b.cols + j_out + 1)) * b.get(b.rows - i + 1, b.cols - j + 1);
                        }
                    }
                    output.set(i_out, j_out, sum);
                }
            }
        } else {
            for(int i_out = 0; i_out < output.rows; i_out++) {
                for(int j_out = 0; j_out < output.cols; j_out++) {
                    double sum = 0;
                    for(int i = 0; i < b.rows; i++) {
                        for(int j = 0; j < b.cols; j++) {
                            sum += this.get(i + i_out, j + j_out) * b.get(i, j);
                        }
                    }
                    output.set(i_out, j_out, sum);
                }
            }
        }

        return output;
    }

    // Testing Matrix things
    public static void main(String[] args) {
        Matrix a = new Matrix(4, 4);
        Matrix b = new Matrix(3, 3);
        a.set(Arrays.asList(
            Arrays.asList(1.0,  2.0,  3.0,  4.0),
            Arrays.asList(5.0,  6.0,  7.0,  8.0),
            Arrays.asList(9.0,  10.0, 11.0, 12.0),
            Arrays.asList(13.0, 14.0, 15.0, 16.0)
        ));
        b.set(Arrays.asList(
            Arrays.asList(1.0, 2.0, 3.0),
            Arrays.asList(4.0, 5.0, 6.0),
            Arrays.asList(7.0, 8.0, 9.0)
        ));
        try {
            Matrix c = a.crossCorrelation(b, true);
            Matrix d = new Matrix(c.rows, c.cols);
            d.set(Arrays.asList(
                Arrays.asList(9.0,   26.0,  50.0,  74.0,  53.0,  28.0),
                Arrays.asList(51.0,  111.0, 178.0, 217.0, 145.0, 72.0),
                Arrays.asList(114.0, 231.0, 348.0, 393.0, 252.0, 120.0),
                Arrays.asList(186.0, 363.0, 528.0, 573.0, 360.0, 168.0),
                Arrays.asList(105.0, 197.0, 274.0, 295.0, 175.0, 76.0),
                Arrays.asList(39.0,  68.0,  86.0,  92.0,  47.0,  16.0)
            ));
            for(int i = 0; i < c.rows; i++) {
                for(int j = 0; j < c.cols; j++) {
                    if(c.get(i,j) != d.get(i,j)) {
                        System.out.println("196: Expected value " + d.get(i,j) + " at (" + i + "," + j + ") but got " + c.get(i,j));
                    }
                }
            }

            c = a.crossCorrelation(b, false);
            d = new Matrix(c.rows, c.cols);
            d.set(Arrays.asList(
                Arrays.asList(348.0,393.0),
                Arrays.asList(528.0,573.0)
            ));
            for(int i = 0; i < c.rows; i++) {
                for(int j = 0; j < c.cols; j++) {
                    if(c.get(i,j) != d.get(i,j)) {
                        System.out.println("210: Expected value " + d.get(i,j) + " at (" + i + "," + j + ") but got " + c.get(i,j));
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
