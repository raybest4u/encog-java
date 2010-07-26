package org.encog.examples.neural.benchmark;

import org.encog.Encog;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.benchmark.RandomTrainingFactory;
import org.encog.util.simple.EncogUtility;

public class StressTest {
	
	public static final int INPUT_COUNT = 100;
	public static final int HIDDEN1_COUNT = 50;
	public static final int HIDDEN2_COUNT = 50;
	public static final int OUTPUT_COUNT = 100;
	public static final int TRAINING_SIZE = 5000;
	
	public static void main(String args[])
	{
		//Encog.getInstance().initCL();
		
		NeuralDataSet trainingSet = RandomTrainingFactory.generate(
				TRAINING_SIZE,
				INPUT_COUNT,
				OUTPUT_COUNT,
				-1,
				1); 
		BasicNetwork network = EncogUtility.simpleFeedForward(INPUT_COUNT, HIDDEN1_COUNT, HIDDEN2_COUNT, OUTPUT_COUNT, true);
		
		EncogUtility.trainToError(network, trainingSet, 0.01);
	}
}
