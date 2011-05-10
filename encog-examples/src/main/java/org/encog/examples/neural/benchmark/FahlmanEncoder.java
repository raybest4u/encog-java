package org.encog.examples.neural.benchmark;

import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.util.simple.EncogUtility;

/**
 * This example implements a Fahlman Encoder.  Though probably not invented by Scott 
 * Fahlman, such encoders were used in many of his papers, particularly:
 * 
 * "An Empirical Study of Learning Speed in Backpropagation Networks" 
 * (Fahlman,1988)
 * 
 * It provides a very simple way of evaluating classification neural networks.
 *   Basically, the input and output neurons are the same in count.  However, 
 *   there is a smaller number of hidden neurons.  This forces the neural 
 *   network to learn to encode the patterns from the input neurons to a 
 *   smaller vector size, only to be expanded again to the outputs.
 * 
 * The training data is exactly the size of the input/output neuron count.  
 * Each training element will have a single column set to 1 and all other 
 * columns set to zero.  You can also perform in "complement mode", where 
 * the opposite is true.  In "complement mode" all columns are set to 1, 
 * except for one column that is 0.  The data produced in "complement mode" 
 * is more difficult to train.
 * 
 * Fahlman used this simple training data to benchmark neural networks when 
 * he introduced the Quickprop algorithm in the above paper.
 *
 */
public class FahlmanEncoder {
	public static final int INPUT_OUTPUT_COUNT = 8;
	public static final int HIDDEN_COUNT = 2;
	public static final boolean COMPL = false;

	public static MLDataSet generateTraining(int inputCount, boolean compl) {
		double[][] input = new double[INPUT_OUTPUT_COUNT][INPUT_OUTPUT_COUNT];
		double[][] ideal = new double[INPUT_OUTPUT_COUNT][INPUT_OUTPUT_COUNT];

		for (int i = 0; i < inputCount; i++) {
			for (int j = 0; j < inputCount; j++) {
				if (compl) {
					input[i][j] = (j == i) ? 0.0 : 1.0;
				} else {
					input[i][j] = (j == i) ? 1.0 : 0.0;
				}

				ideal[i][j] = input[i][j];
			}
		}

		return new BasicMLDataSet(input, ideal);
	}

	public static void main(String[] args) {
		MLDataSet trainingData = generateTraining(INPUT_OUTPUT_COUNT, COMPL);
		MLMethod method = EncogUtility.simpleFeedForward(INPUT_OUTPUT_COUNT,
				HIDDEN_COUNT, 0, INPUT_OUTPUT_COUNT, false);
		EncogUtility.trainToError(method, trainingData, 0.01);
		EncogUtility.evaluate((MLRegression) method, trainingData);
	}
}
