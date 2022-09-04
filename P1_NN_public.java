// Written by: Wai Zin Linn
// Attribution: Young Wu and Hugh Liu's CS540 P1 Solution 2020

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class P1_NN_public {

	// TODO: change hyper-parameters below, like MAX_EPOCHS, alpha, etc.
	private static final String COMMA_DELIMITER = ",";
	private static final String PATH_TO_TRAIN = "./mnist_train.csv";
	private static final String NEW_TEST = "./test.txt";
	private static final int MAX_EPOCHS = 30;
	private static final double ALPHA = 0.001111;
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
					continue;
				if (LABELS[0].equals(string_values[0]))
					records[k][0] = 0.0; // label 0
				else
					records[k][0] = 1.0; // label 1
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

	public static double[][] newTest(String file_path) throws FileNotFoundException, IOException {
		double[][] records = new double[200][784];
		try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
			String line;
			int k = 0;
			while ((line = br.readLine()) != null) {
				String[] string_values = line.split(COMMA_DELIMITER);
				for (int i = 0; i < string_values.length; i++)
					records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
				k += 1;
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

	public static double diff_sigmoid(double o) {
		return o * (1 - o);
	}

	public static void main(String[] args) throws IOException {
		double[][] train = parseRecords(PATH_TO_TRAIN);
		int m = 784; // number of inputs
		int h = 28; // number of hidden units
		double[][] w1 = new double[m][h]; //first layer of weight
		double[] w2 = new double[h]; //second layer of weight
		double[] b1 = new double[h];
		double b2 = 2 * rng.nextDouble() - 1;
		int num_train = train.length;
		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[0].length; j++)
				w1[i][j] = 2 * rng.nextDouble() - 1; // initialize weights
		}
		for (int i = 0; i < w2.length; i++)
			w2[i] = 2 * rng.nextDouble() - 1; // initialize weights
		for (int i = 0; i < b1.length; i++)
			b1[i] = 2 * rng.nextDouble() - 1;
		double loss_prev = 1.0e10;
		for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
			List<double[]> list = Arrays.asList(train); // Shuffle the data set
			Collections.shuffle(list);
			train = list.toArray(train);
			double[][] a1 = new double[num_train][h];
			double[] a2 = new double[num_train];
			for (int ind = 0; ind < num_train; ind++) {
				// calculate out_h
				for (int i = 0; i < h; i++) {
					double s = 0.0;
					for (int j = 0; j < m; j++)
						s += w1[j][i] * train[ind][j + 1];
					a1[ind][i] = sigmoid(s + b1[i]);
				}
				// calculate out_o
				double s = 0.0;
				for (int i = 0; i < h; i++)
					s += a1[ind][i] * w2[i];
				a2[ind] = sigmoid(s + b2);
				double label = train[ind][0];
				// update w1
				for (int i = 0; i < h; i++) {
					for (int j = 0; j < m; j++)
						w1[j][i] -= ALPHA * (a2[ind] - label) * diff_sigmoid(a2[ind]) * w2[i]
								* diff_sigmoid(a1[ind][i]) * train[ind][j + 1];
				}
				// update w2
				for (int i = 0; i < h; i++)
					w2[i] -= ALPHA * (a2[ind] - label) * diff_sigmoid(a2[ind]) * a1[ind][i];
				// update b1
				for (int i = 0; i < h; i++)
					b1[i] -= ALPHA * (a2[ind] - label) * diff_sigmoid(a2[ind]) * w2[i]
							* diff_sigmoid(a1[ind][i]);
				// update b2
				b2 -= ALPHA * (a2[ind] - label) * diff_sigmoid(a2[ind]);
			}
			// calculate error
			double loss = 0;
			for (int ind = 0; ind < num_train; ind++)
				loss += 0.5 * (train[ind][0] - a2[ind]) * (train[ind][0] - a2[ind]);
			double loss_reduction = loss_prev - loss;
			loss_prev = loss;
			// count correct
			double correct = 0.0;
			for (int ind = 0; ind < num_train; ind++) {
				if ((train[ind][0] == 1.0 && a2[ind] >= 0.5) || (train[ind][0] == 0.0 && a2[ind] < 0.5))
					correct += 1.0;
			}
			double acc = correct / num_train;
			System.out.println("epoch = " + epoch + ", loss = " + loss + ", loss reduction = " + loss_reduction
					+ ", correctly classified = " + acc);
		}
		
		double[][] test = newTest(NEW_TEST);
		
		//Q5
		File result = new File("q5.txt");
		FileWriter write = new FileWriter(result);
		PrintWriter pw = new PrintWriter(write);

		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[i].length; j++) {
				if (j != w1[i].length - 1) {
					pw.write(Math.round(w1[i][j] * 10000.0) / 10000.0 + ",");
				} else {
					pw.write("" + Math.round(w1[i][j] * 10000.0) / 10000.0);
				}
			}
			pw.write("\n");
		}

		for (int g = 0; g < b1.length; g++) {
			if (g != b1.length - 1) {
				pw.write(Math.round(b1[g] * 10000.0) / 10000.0 + ",");
			} else {
				pw.write("" + Math.round(b1[g] * 10000.0) / 10000.0);
			}
		}

		pw.close();

		// Q6
		File result1 = new File("q6.txt");
		FileWriter write1 = new FileWriter(result1);
		PrintWriter pw1 = new PrintWriter(write1);
		for (int r = 0; r < w2.length; r++) {
				pw1.write(Math.round(w2[r]* 10000.0) / 10000.0 + ",");
		}

		pw1.write("" + Math.round(b2* 10000.0) / 10000.0);
		pw1.close();
		
		//Q7 and Q8
		File result2 = new File("q7.txt");
		FileWriter write2 = new FileWriter(result2);
		PrintWriter pw2 = new PrintWriter(write2);
		File result3 = new File("q8.txt");
		FileWriter write3 = new FileWriter(result3);
		PrintWriter pw3 = new PrintWriter(write3);
		
		double[][] testArr = new double[test.length][h];
		double[] testResult = new double[test.length];
		
		for (int ind = 0; ind < test.length; ind++) {
			// calculate out_h for test
			for (int i = 0; i < h; i++) {
				double s = 0.0;
				for (int j = 0; j < m; j++)
					s += w1[j][i] * test[ind][j];
				testArr[ind][i] = sigmoid(s + b1[i]);
			}
			// calculate out_o for test
			double s = 0.0;
			for (int i = 0; i < h; i++)
				s += testArr[ind][i] * w2[i];
			testResult[ind] = sigmoid(s + b2);
			pw2.write(Math.round(testResult[ind]*100.0)/100.0 + ",");
			pw3.write(Math.round(testResult[ind]) + ",");
		}
		
		pw2.close();
		pw3.close();
		
		//Q9
		File result4 = new File("q9.txt");
		FileWriter write4 = new FileWriter(result4);
		PrintWriter pw4 = new PrintWriter(write4);
		for(int o = 0; o < 1; o++) {
			for(int y = 0; y < test[o].length; y++) {
				pw4.write(Math.round(test[o][y]) + ",");
			}
		}
		pw4.close();
		
		System.out.println(test.length > 0);
	}
}