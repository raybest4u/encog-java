package org.encog.examples.neural.recurrant.elman;



import org.encog.bot.browse.extract.ListExtractListener;
import org.encog.examples.neural.util.TemporalXOR;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.layers.ContextLayer;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.synapse.SynapseType;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.strategy.Greedy;
import org.encog.neural.networks.training.strategy.HybridStrategy;
import org.encog.neural.networks.training.strategy.ResetStrategy;
import org.encog.neural.networks.training.strategy.SmartLearningRate;
import org.encog.neural.networks.training.strategy.StopTrainingStrategy;
import org.encog.util.Logging;
import org.encog.util.randomize.FanInRandomizer;
import org.encog.util.randomize.RangeRandomizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ElmanXOR {
	
	static BasicNetwork createElmanNetwork()
	{
		// construct an Elman type network
		Layer hidden;
		Layer context = new ContextLayer(2);
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(1));
		network.addLayer(hidden = new BasicLayer(2));
		hidden.addNext(context,SynapseType.OneToOne);
		context.addNext(hidden);
		network.addLayer(new BasicLayer(1));
		network.getStructure().finalizeStructure();
		//network.reset();
		(new RangeRandomizer(-1.0,1.0)).randomize(network);
		return network;
	}
	
	static BasicNetwork createFeedforwardNetwork()
	{
		// construct a feedforward type network

		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(1));
		network.addLayer(new BasicLayer(2));
		network.addLayer(new BasicLayer(1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	public static double trainNetwork(String what, BasicNetwork network,NeuralDataSet trainingSet)
	{
		// train the neural network
		final NeuralSimulatedAnnealing trainAlt = new NeuralSimulatedAnnealing(
				network, trainingSet, 10, 2, 100);
		
		final Train trainMain = new Backpropagation(network, trainingSet, 0.00001, 0.0);

		StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);

		int epoch = 0;
		while(!stop.shouldStop()) {
			trainMain.iteration();
			System.out.println("Training "+what+", Epoch #" + epoch + " Error:" + trainMain.getError());
			epoch++;
		} 	
		return trainMain.getError();
	}
	
	public static void main(String args[])
	{
		Logging.stopConsoleLogging();
		TemporalXOR temp = new TemporalXOR();
		NeuralDataSet trainingSet = temp.generate(100);
		
		BasicNetwork elmanNetwork = createElmanNetwork();
		BasicNetwork feedforwardNetwork = createFeedforwardNetwork();
		
		double elmanError = trainNetwork("Elman",elmanNetwork,trainingSet);
		double feedforwardError = trainNetwork("Feedforward",feedforwardNetwork,trainingSet);
		
		System.out.println("Best error rate with Elman Network: " + elmanError);
		System.out.println("Best error rate with Feedforward Network: " + feedforwardError);
		System.out.println("Elman should be able to get into the 30% range,\nfeedforward should not go below 50%.\nThe recurrent Elment net can learn better in this case.");
		System.out.println("If your results are not as good, try rerunning, or perhaps training longer.");
	}
}
