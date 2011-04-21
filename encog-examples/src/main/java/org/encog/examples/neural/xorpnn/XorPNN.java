package org.encog.examples.neural.xorpnn;

import java.util.Arrays;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.training.pnn.TrainBasicPNN;
import org.encog.neural.pnn.BasicPNN;
import org.encog.neural.pnn.PNNKernelType;
import org.encog.neural.pnn.PNNOutputMode;
import org.encog.util.logging.Logging;

public class XorPNN {

	public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
			{ 0.0, 1.0 }, { 1.0, 1.0 } };

	public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

	
	public static void evaluate( BasicPNN network, BasicMLDataSet trainingSet)
	{
		for (final MLDataPair pair : trainingSet) {
			final MLData output = network.compute(pair.getInput());
			
			Arrays.toString(output.getData());
			System.out.print("Input: ");
			System.out.print(Arrays.toString(pair.getInput().getData()));
			
			if( pair.isSupervised() ) {
				System.out.print(", Ideal: ");
				System.out.print(Arrays.toString(pair.getIdeal().getData()));
			}
			
			System.out.print(", Actual: ");
			System.out.println(Arrays.toString(output.getData()));
		}		
	}
	
	public static void regressionExample()
	{
		System.out.println();
		System.out.println("** Running in regression mode **");
		
		PNNOutputMode mode = PNNOutputMode.Regression;

		BasicPNN network = new BasicPNN(PNNKernelType.Gaussian, mode, 2, 1);

		BasicMLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT,
				XOR_IDEAL);

		System.out.println("Learning...");

		TrainBasicPNN train = new TrainBasicPNN(network, trainingSet);
		train.iteration();
		evaluate(network,trainingSet);
	}
	
	public static void classificationExample()
	{
		System.out.println();
		System.out.println("** Running in classification mode **");
		
		PNNOutputMode mode = PNNOutputMode.Classification;

		BasicPNN network = new BasicPNN(PNNKernelType.Gaussian, mode, 2, 2);

		BasicMLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT,
				XOR_IDEAL);

		System.out.println("Learning...");

		TrainBasicPNN train = new TrainBasicPNN(network, trainingSet);
		train.iteration();
		evaluate(network,trainingSet);
	}
	
	
	public static void main(final String args[]) {

		Logging.stopConsoleLogging();
		regressionExample();
		classificationExample();
	}
}
