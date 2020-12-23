import org.jblas.DoubleMatrix;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

import static java.lang.Math.round;
import static java.lang.Math.sqrt;


//////////////////////////////////////////////////////////////////////////////////////
// Лабораторная работа 2 по дисциплине МРЗвИС
// Выполнена студентом группы 821701
// БГУИР Поживилко Петром Сергеевичем
// Файл main с функциями для обучения, генерации весов и пересчета весов
// 10.12.2020 ver. 0.2
//
// библиотека для работы с матрицами http://jblas.org/
// Авторство Поживилко Петр Сергеевич





//
public class Lab2 {



    public static void main(String[] args) throws IOException {
    double[] learnData1 = {0.01,0.02,0.03,0.04,0.05,0.06,0.07};
    double[] testData1={0.02,0.03,0.04,0.05,0.06,0.07};

    DoubleMatrix testVector=new DoubleMatrix(testData1).transpose();
    int inputLength=learnData1.length;
    int neuronCount=5;

    boolean zero=true;





    DoubleMatrix learnVector1=new DoubleMatrix(1,inputLength-1);
    for(int i=0;i<inputLength-1;i++){
        learnVector1.put(0,i,learnData1[i]);
    }
    DoubleMatrix lastElement=new DoubleMatrix(1,1);
    lastElement.put(0,0,learnData1[inputLength-1]);
    DoubleMatrix weights1=generateWeights(inputLength+neuronCount,neuronCount);
    DoubleMatrix weights2=generateWeights(neuronCount,1);
    //DoubleMatrix weights1=generateWeights(inputLength-1,neuronCount);
     double alpha=0.01;

     double e=0.001;
     double E=10000;

        DoubleMatrix elman = new DoubleMatrix(1,neuronCount);
        DoubleMatrix jordan = new DoubleMatrix(1, 1);
        learn(learnVector1,weights1,weights2,e,elman,jordan,zero,lastElement,alpha);

        //System.out.println(Arrays.toString(testData1));
        //testVector.mmul(weights1).mmul(weights2).print();







    }
    public static void learn(DoubleMatrix learnVector,
                             DoubleMatrix weights1,
                             DoubleMatrix weights2,
                             double e,
                             DoubleMatrix elman,
                             DoubleMatrix jordan,
                             boolean zero,
                             DoubleMatrix lastElement,
                             double alpha) {
        DoubleMatrix nullMatrixElman = new DoubleMatrix(1, weights2.rows);
        DoubleMatrix nullMatrixJordan = new DoubleMatrix(1, 1);
        int counter = 0;
        double E = 10000;
        while (E > e) {
            counter++;
            DoubleMatrix input = addContext(learnVector, elman, jordan);
            E = 0.;
            DoubleMatrix firstLayer = input.mmul(weights1);
            DoubleMatrix firstLayerAfterActivation = activate(firstLayer);
            DoubleMatrix secondLayer = firstLayer.mmul(weights2);
            DoubleMatrix secondLayerAfterActivation = activate(secondLayer);
            if (zero) {
                elman = nullMatrixElman;
                jordan = nullMatrixJordan;
            } else {
                elman = fillContext(firstLayerAfterActivation);
                jordan = fillContext(secondLayerAfterActivation);
            }

            //change weights1
            weights1.subi((changeW1(lastElement, secondLayerAfterActivation, secondLayer, firstLayerAfterActivation, input, weights2, weights1)).mmul(alpha));

            E = Math.abs(lastElement.sub(secondLayer).get(0, 0));
            //change weights2
            weights2.subi(((changeW2(lastElement, secondLayerAfterActivation, secondLayer, firstLayer))).mmul(alpha));

            System.out.println(E);
        }
        System.out.println(counter);
        predict(learnVector,weights1,weights2,elman,jordan);
    }
    

    public static void predict(DoubleMatrix input,DoubleMatrix w1,DoubleMatrix w2,DoubleMatrix elman,DoubleMatrix jordan){
        input.print();
        DoubleMatrix inputWithContext = addContext(input, elman, jordan);
        DoubleMatrix firstLayer = inputWithContext.mmul(w1);
        firstLayer = activate(firstLayer);
        DoubleMatrix secondLayer = firstLayer.mmul(w2);
        secondLayer = activate(secondLayer);
        System.out.println("next value");
        secondLayer.print();


    }

    public static DoubleMatrix activate(DoubleMatrix vector){
        DoubleMatrix answer = new DoubleMatrix(1, vector.columns);
        for(int i=0;i<vector.columns;i++){
            answer.put(0,i,Math.tanh(vector.get(0, i)));
        }
        return answer;
    }

    public static DoubleMatrix addContext(DoubleMatrix inputVector,DoubleMatrix elman,DoubleMatrix jordan){
        ArrayList<Double> answer=new ArrayList<Double>();
        for(int i=0;i<inputVector.columns;i++){
            answer.add(inputVector.get(0, i));
        }
        for(int i=0;i<elman.columns;i++){
            answer.add(elman.get(0, i));
        }

            answer.add(jordan.get(0, 0));

        return new DoubleMatrix(answer).transpose();
    }





    public static DoubleMatrix generateWeights(int input,int output){
        double[][] weights = new double[input][output];
        Random random = new Random();
        for(int i=0;i<input;i++){
            for(int j=0;j<output;j++){

                weights[i][j] = (random.nextDouble());
            }
        }
        return new DoubleMatrix(weights);
    }
    public static double findError(DoubleMatrix delta){
        double answer=0;
        for(int i=0;i<delta.length;i++){
            answer += delta.get(i) * delta.get(i);
        }
        return answer;
    }




    public static DoubleMatrix fillContext(DoubleMatrix vector){
        DoubleMatrix answer = new DoubleMatrix(1,vector.columns);
        for(int i=0;i<vector.columns;i++){
            answer.put(0,i,vector.get(0, i));
        }
        return answer;
    }


    public static void normalizeMatrix(DoubleMatrix matrix) {
        int cols = matrix.columns;
        int rows = matrix.rows;
        for (int i = 0; i < cols; i ++) {
            double sum = 0.;
            for (int j = 0; j < rows; j++) {
                sum += (matrix.get(j, i) * matrix.get(j, i));
            }
            sum = sqrt(sum);
            for (int j = 0; j < rows; j++) {
                double newValue = matrix.get(j, i) / sum;
                matrix.put(j, i, newValue);
            }
        }
    }

    public static double dF(double x){
        return 1./(Math.cosh(x)*Math.cosh(x));
    }



    //Author Бутрин Станислав
    public static DoubleMatrix changeW2(DoubleMatrix etalon,DoubleMatrix second_out,DoubleMatrix second_net,DoubleMatrix first_out){
        DoubleMatrix der_v = new DoubleMatrix(first_out.columns,1);
        for(int i=0;i<der_v.rows;i++){
        double der_e_der_out = second_out.sub(etalon).get(0,0);
        double der_out_der_net = dF(second_net.get(0,0));
        double der_net_der_v = first_out.get(0,i);
        double der_e_der_v = der_e_der_out * der_out_der_net * der_net_der_v;
        der_v.put(i,0,der_e_der_v);

        }
        return der_v;
    }

    public static DoubleMatrix changeW1(DoubleMatrix etalon,
                                        DoubleMatrix second_out,
                                        DoubleMatrix second_net,
                                        DoubleMatrix first_net,
                                        DoubleMatrix inputs,
                                        DoubleMatrix weights2,
                                        DoubleMatrix weights1){
        DoubleMatrix der_w = new DoubleMatrix(weights1.rows, weights1.columns);
        for(int i=0;i<der_w.rows;i++){
            for(int j=0;j<der_w.columns;j++){
                double der_e_der_out = second_out.sub(etalon).get(0, 0);
                double der_out_der_net=dF(second_net.get(0,0));
                double  der_net_der_out_w=weights2.get(j,0);
                double der_out_w_der_net_w=dF(first_net.get(0,j));
                double der_net_w_der_w = inputs.get(0, i);
                double der_w_i_j = der_e_der_out * der_out_der_net * der_net_der_out_w * der_out_w_der_net_w * der_net_w_der_w;
                der_w.put(i, j, der_w_i_j);
            }
        }
        return der_w;
    }














    




}
