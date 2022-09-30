package nn;

import java.util.Arrays;

public class TestMatrix {
    public static void main(String[] args) throws Exception{
        Matrix a = new Matrix(3, 3);
        a.values = Arrays.asList(Arrays.asList(1.0, 2.0, 3.0), Arrays.asList(4.0, 5.0, 6.0), Arrays.asList(7.0, 8.0, 9.0));
        Matrix b = new Matrix(3, 2);
        b.values = Arrays.asList(Arrays.asList(1.0, 2.0), Arrays.asList(4.0, 5.0), Arrays.asList(7.0, 8.0));
        if(!a.multiply(b).values.equals(Arrays.asList(Arrays.asList(30.0, 36.0), Arrays.asList(66.0, 81.0), Arrays.asList(102.0, 126.0)))) {
            throw new Exception("Failing multiplication test");
        }
        b.values = Arrays.asList(Arrays.asList(1.0, 2.0, 3.0), Arrays.asList(4.0, 5.0, 6.0), Arrays.asList(7.0, 8.0, 9.0));
        b.cols = 3;
        if(!a.add(b).values.equals(Arrays.asList(Arrays.asList(2.0, 4.0, 6.0), Arrays.asList(8.0, 10.0, 12.0), Arrays.asList(14.0, 16.0, 18.0)))) {
            throw new Exception("Failing addition test");
        }
    }
}
