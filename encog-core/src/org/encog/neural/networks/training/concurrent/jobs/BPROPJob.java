package org.encog.neural.networks.training.concurrent.jobs;

import org.encog.engine.opencl.EncogCLDevice;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Strategy;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class BPROPJob extends TrainingJob {

	private double learningRate;
	private double momentum;
	
	
	public BPROPJob(BasicNetwork network, NeuralDataSet training, boolean loadToMemory,
			double learningRate, double momentum) {
		super(network,training,loadToMemory);
		this.learningRate = learningRate;
		this.momentum = momentum;
	}


	/**
	 * @return the learningRate
	 */
	public double getLearningRate() {
		return learningRate;
	}


	/**
	 * @param learningRate the learningRate to set
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}


	/**
	 * @return the momentum
	 */
	public double getMomentum() {
		return momentum;
	}


	/**
	 * @param momentum the momentum to set
	 */
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}


	@Override
	public void createTrainer(EncogCLDevice device) {
		Train train = new Backpropagation(
				getNetwork(),
				getTraining(),
				device,
				getLearningRate(),
				getMomentum());
		
		for(Strategy strategy : this.getStrategies() ) {
			train.addStrategy(strategy);
		}
		
		this.setTrain(train);
	}
	
	
	
}
