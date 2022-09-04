// Written by: Wai Zin Linn
// Attribution: Young Wu and Hugh Liu's CS540 P1 Solution 2020

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

public class P1_LR_public {
	// TODO: change hyper-parameters HERE, like iterations, learning_rate, etc.
	private static final String COMMA_DELIMITER = ",";
	private static final String PATH_TO_TRAIN = "./mnist_train.csv";
	private static final String NEW_TEST = "./test.txt";
	private static final int MAX_EPOCHS = 20; //
	private static final double ALPHA = 0.0001111; //learning rate
	private static final String[] LABELS = new String[] { "6", "9" };
	private static Random rng = new Random();

	public static double[][] parseRecords(String file_path) throws FileNotFoundException, IOException {
		double[][] records = new double[20000][786];
		try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
			String line;
			int k = 0;
			while ((line = br.readLine()) != null) {
				String[] string_values = line.split(COMMA_DELIMITER);
				if (!string_values[0].equals(LABELS[0]) && !string_values[0].contentEquals(LABELS[1]))
					continue; //skipping the numbers if not equal to 6 or 9
				if (LABELS[0].equals(string_values[0]))
					records[k][0] = 0.0; // equal to 6 -> 0 
				else
					records[k][0] = 1.0; // equal to 9 -> 1
				for (int i = 1; i < string_values.length; i++)
					records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
				k += 1;
			}
			double[][] res = new double[k][786];
			for (int i = 0; i < k; i++)
				System.arraycopy(records[i], 0, res[i], 0, 786);
			return res;
		}
	}

	public static double[][] NewTest(String file_path) throws FileNotFoundException, IOException {
		double[][] records = new double[200][784];
		try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
			String line;
			int k = 0;
			while ((line = br.readLine()) != null) {
				String[] string_values = line.split(COMMA_DELIMITER);
				for (int i = 0; i < string_values.length; i++)
					records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
				k++;
			}
			double[][] res = new double[k][784];
			for (int i = 0; i < k; i++)
				System.arraycopy(records[i], 0, res[i], 0, 784);
			return res;
		}
	}

	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	public static void main(String[] args) throws IOException {
		double[][] train = parseRecords(PATH_TO_TRAIN);
		
		//Answer for q1
		File result1 = new File("q1.txt");
		FileWriter write = new FileWriter(result1);
		PrintWriter pw = new PrintWriter(write);
		for(int d = 0; d < 1; d++) {
			for(int e = 0; e < 784; e++) {
				if(e != 783) {
					pw.write(Math.round(train[d][e]) + ",");
				}else {
					pw.write("" + (int) Math.round(train[d][e]));
				}
			}
			pw.close();
		}
		
		
		int m = 784; // number of inputs
		double[] w = new double[m];
		double b = 2 * rng.nextDouble() - 1;
		
		for (int i = 0; i < w.length; i++)
			w[i] = 2 * rng.nextDouble() - 1; // initialize weights

		

		int num_train = train.length;
		
		double loss_prev = 0.0;
		
		
		//Graident Descent step
		for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
			// calculate a_i array
			double[] a = new double[num_train];
			for (int ind = 0; ind < num_train; ind++) {
				double s = 0;
				for (int i = 0; i < w.length; i++)
					s += w[i] * train[ind][i + 1];
				a[ind] = sigmoid(s + b);
			}
			
			// update weights
			for (int j = 0; j < w.length; j++) {
				double dw = 0.0;
				for (int i = 0; i < num_train; i++)
					dw += (a[i] - train[i][0]) * train[i][j + 1];
				w[j] -= ALPHA * dw;
			}
			
			// update bias
			double db = 0;
			for (int i = 0; i < num_train; i++)
				db += (a[i] - train[i][0]);
			b -= ALPHA * db;
			
			// calculate loss
			double loss = 0.0;
			for (int i = 0; i < num_train; i++) {
				if (train[i][0] == 0.0) {
					if (a[i] > 0.9999)
						loss += 100.0; // something large
					else
						loss -= Math.log(1 - a[i]);
				} else if (train[i][0] == 1.0) {
					if (a[i] < 0.0001)
						loss += 100.0;
					else
						loss -= Math.log(a[i]);
				}
			}
			double loss_reduction = loss_prev - loss;
			loss_prev = loss;
			
			// count correct
			double correct = 0.0;
			for (int ind = 0; ind < num_train; ind++) {
				if ((train[ind][0] == 1.0 && a[ind] >= 0.5) || (train[ind][0] == 0.0 && a[ind] < 0.5))
					correct += 1.0;
			}
			double acc = correct / num_train;
			System.out.println("epoch = " + epoch + ", loss = " + loss + ", loss reduction = " + loss_reduction
					+ ", correctly classified = " + acc);
			
		}
		
		//Answer for q2
		File result2 = new File("q2.txt");
		FileWriter write1 = new FileWriter(result2);
	    PrintWriter pw1 = new PrintWriter(write1);
				
		for (int h = 0; h < w.length; h++) {
			pw1.write(Math.round(w[h] * 10000.0)/10000.0 + ",");
		}
				
		pw1.write("" + Math.round(b*10000.0)/10000.0);
		pw1.close();
				
		double[][] test = NewTest(NEW_TEST);
		//calculate for test array
		//Answer for q3
		File result3 = new File("q3.txt");
		File result4 = new File("q4.txt");
		FileWriter write2 = new FileWriter(result3);
		FileWriter write3 = new FileWriter(result4);
		PrintWriter pw2 = new PrintWriter(write2);
		PrintWriter pw3 = new PrintWriter(write3);
		double[] testArr = new double[test.length];
		for (int ind = 0; ind < test.length; ind++) {
			double testResult = 0;
			for (int i = 0; i < w.length; i++)
				testResult += w[i] * test[ind][i]; //removed +1 because there is a header on training set
			testArr[ind] = sigmoid(testResult + b);
			
			//Answer for q4
			
			if(testArr[ind] < 0.5) {
				pw3.write(0 + ",");
			}else {
				pw3.write(1 + ",");
			}
			pw2.write(Math.round(testArr[ind]) + ",");
		}
		
		pw2.close();
		pw3.close();
	}
}