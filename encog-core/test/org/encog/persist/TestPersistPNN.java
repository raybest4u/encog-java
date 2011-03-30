package org.encog.persist;

import java.io.File;
import java.io.IOException;

import org.encog.neural.art.ART1;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.XOR;
import org.encog.neural.networks.training.pnn.TrainBasicPNN;
import org.encog.neural.pnn.BasicPNN;
import org.encog.neural.pnn.PNNKernelType;
import org.encog.neural.pnn.PNNOutputMode;
import org.encog.util.obj.SerializeObject;

import junit.framework.TestCase;

public class TestPersistPNN extends TestCase {
	
	public final String EG_FILENAME = "encogtest.eg";
	public final String SERIAL_FILENAME = "encogtest.ser";
	
	public BasicPNN create() {
		PNNOutputMode mode = PNNOutputMode.Regression;

		BasicPNN network = new BasicPNN(PNNKernelType.Gaussian, mode, 2, 1);

		BasicNeuralDataSet trainingSet = new BasicNeuralDataSet(XOR.XOR_INPUT,
				XOR.XOR_IDEAL);

		System.out.println("Learning...");

		TrainBasicPNN train = new TrainBasicPNN(network, trainingSet);
		train.learn();
		XOR.verifyXOR(network, 0.001);
		return network;		
	}
	
	public void testPersistEG()
	{
		BasicPNN network = create();

		EncogDirectoryPersistence.saveObject(new File(EG_FILENAME), network);
		BasicPNN network2 = (BasicPNN)EncogDirectoryPersistence.loadObject(new File(EG_FILENAME));

		XOR.verifyXOR(network2, 0.001);
	}
	
	public void testPersistSerial() throws IOException, ClassNotFoundException
	{
		BasicPNN network = create();
		
		SerializeObject.save(SERIAL_FILENAME, network);
		BasicPNN network2 = (BasicPNN)SerializeObject.load(SERIAL_FILENAME);
				
		XOR.verifyXOR(network2, 0.001);
	}
}
