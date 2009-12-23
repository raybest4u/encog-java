package org.encog.neural.networks.training;

import java.util.Iterator;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.NetworkUtil;
import org.encog.neural.networks.XOR;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.genetic.NeuralGeneticAlgorithm;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.encog.neural.prune.PruneSelective;
import org.encog.util.logging.Logging;
import org.encog.util.randomize.RangeRandomizer;
import org.junit.Test;

public class TestTraining extends TestCase   {

	
	@Test
	public void testRPROP() throws Throwable
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);
		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		Train rprop = new ResilientPropagation(network, trainingData);
		NetworkUtil.testTraining(rprop,0.03);
	}
	
	@Test
	public void testBPROP() throws Throwable
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);
		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		Train bprop = new Backpropagation(network, trainingData, 0.7, 0.9);
		NetworkUtil.testTraining(bprop,0.01);
	}
	
	@Test
	public void testManhattan() throws Throwable
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);
		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		Train bprop = new ManhattanPropagation(network, trainingData, 0.01);
		NetworkUtil.testTraining(bprop,0.01);
	}
	
	@Test
	public void testSCG() throws Throwable
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);
		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		Train bprop = new ScaledConjugateGradient(network, trainingData);
		NetworkUtil.testTraining(bprop,0.04);
	}
	
	@Test
	public void testAnneal() throws Throwable
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		CalculateScore score = new TrainingSetScore(trainingData);
		NeuralSimulatedAnnealing anneal = new NeuralSimulatedAnnealing(network,score,10,2,100);
		NetworkUtil.testTraining(anneal,0.01);
	}
	
	@Test
	public void testGenetic() throws Throwable
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		CalculateScore score = new TrainingSetScore(trainingData);
		NeuralGeneticAlgorithm genetic = new NeuralGeneticAlgorithm(network, new RangeRandomizer(-1,1), score, 500,0.1,0.25);
		NetworkUtil.testTraining(genetic,0.00001);
	}
	
	public void testCont()
	{
		Logging.stopConsoleLogging();
		NeuralDataSet trainingData = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);
		
		BasicNetwork network = NetworkUtil.createXORNetworkUntrained();
		Propagation prop = new Backpropagation(network, trainingData, 0.7, 0.9);
		
		Assert.assertFalse(prop.canContinue());
		
		try
		{
			prop.pause();
			Assert.assertFalse(true);
		}
		catch(Exception e)
		{
			// we want an exception.
		}
		
		try
		{
			prop.resume(null);
			Assert.assertFalse(true);
		}
		catch(Exception e)
		{
			// we want an exception.
		}
	}
	


}
