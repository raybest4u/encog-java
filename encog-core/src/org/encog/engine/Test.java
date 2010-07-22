package org.encog.engine;

import org.encog.engine.data.BasicEngineDataSet;
import org.encog.engine.data.EngineDataSet;
import org.encog.engine.network.flat.FlatNetwork;
import org.encog.engine.network.train.TrainFlatNetworkResilient;
import org.encog.engine.data.EngineData;


public class Test {
	
	public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
		{ 0.0, 1.0 }, { 1.0, 1.0 } };

	public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

	public static double[] weight = { 0.4228153432223951,0.02111537373510175,0.5962239679273846,-0.45942499762994826,0.7820765049015583,-0.4758012544007315,-0.2556326473625983,0.01101103692953287,0.8514177374435754,-0.30646946483397164,0.30277558799351034,-0.8663710588956988,0.7644787157572839 };

	
	public static void main(String[] args)
	{
		//EncogEngine.getInstance().initCL();
		FlatNetwork network = new FlatNetwork(2,3,0,1,true);
		
		System.out.println( network.getWeights().length );
		
		/*
		for(int i=0;i<network.getWeights().length;i++)
		{
			network.getWeights()[i] = (Math.random()*2.0) - 1.0;
		}
		*/
		
		network.decodeNetwork(weight);	 
		
		EngineDataSet trainingSet = new BasicEngineDataSet(XOR_INPUT, XOR_IDEAL);
		
		TrainFlatNetworkResilient train = new TrainFlatNetworkResilient(network,trainingSet);
		
		int epoch = 1;

		do {
			train.iteration();
			System.out
					.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		} while(train.getError() > 0.01 );

		// test the neural network
		double[] output = new double[2];
		System.out.println("Neural Network Results:");
		for(EngineData pair: trainingSet ) {
			network.compute(pair.getInput(),output);
			System.out.println(pair.getInput()[0] + "," + pair.getInput()[1]
					+ ", actual=" + output[0] + ",ideal=" + pair.getIdeal()[0]);
		}
	}
}
