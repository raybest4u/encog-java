/*
 * Encog(tm) Artificial Intelligence Framework v2.3
 * Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2010 by Heaton Research Inc.
 * 
 * Released under the LGPL.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 * 
 * Encog and Heaton Research are Trademarks of Heaton Research, Inc.
 * For information on Heaton Research trademarks, visit:
 * 
 * http://www.heatonresearch.com/copyright.html
 */

package org.encog.neural.networks.training.anneal;

import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.structure.NetworkCODEC;
import org.encog.neural.networks.training.BasicTraining;
import org.encog.neural.networks.training.CalculateScore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class implements a simulated annealing training algorithm for 
 * neural networks. It is based on the generic SimulatedAnnealing class.
 * It is used in the same manner as any other training class that implements the
 * Train interface.  There are essentially two ways you can make use of this
 * class.
 * 
 * Either way, you will need a score object.  The score object tells the
 * simulated annealing algorithm how well suited a neural network is.
 * 
 * If you would like to use simulated annealing with a training set you 
 * should make use TrainingSetScore class.  This score object uses a training
 * set to score your neural network.
 * 
 * If you would like to be more abstract, and not use a training set, you
 * can create your own implementation of the CalculateScore method.  This
 * class can then score the networks any way that you like.  
 * 
 */
public class NeuralSimulatedAnnealing extends BasicTraining {

	/**
	 * The cutoff for random data.
	 */
	public static final double CUT = 0.5;

	/**
	 * The logging object.
	 */
	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * The neural network that is to be trained.
	 */
	private final BasicNetwork network;

	/**
	 * This class actually performs the training.
	 */
	private final NeuralSimulatedAnnealingHelper anneal;
	
	/**
	 * Used to calculate the score.
	 */
	private final CalculateScore calculateScore;

	/**
	 * Construct a simulated annleaing trainer for a feedforward neural network.
	 * 
	 * @param network
	 *            The neural network to be trained.
	 * @param calculateScore
	 * 			  Used to calculate the score for a neural network.
	 * @param startTemp
	 *            The starting temperature.
	 * @param stopTemp
	 *            The ending temperature.
	 * @param cycles
	 *            The number of cycles in a training iteration.
	 */
	public NeuralSimulatedAnnealing(final BasicNetwork network,
			final CalculateScore calculateScore,
			final double startTemp,
			final double stopTemp, 
			final int cycles) {
		this.network = network;
		this.calculateScore = calculateScore;
		this.anneal = new NeuralSimulatedAnnealingHelper(this);
		this.anneal.setTemperature(startTemp);
		this.anneal.setStartTemperature(startTemp);
		this.anneal.setStopTemperature(stopTemp);
		this.anneal.setCycles(cycles);
	}

	/**
	 * Get the best network from the training.
	 * 
	 * @return The best network.
	 */
	public BasicNetwork getNetwork() {
		return this.network;
	}

	/**
	 * Perform one iteration of simulated annealing.
	 */
	public void iteration() {
		if (this.logger.isInfoEnabled()) {
			this.logger.info("Performing Simulated Annealing iteration.");
		}
		preIteration();
		this.anneal.iteration();
		setError(this.anneal.calculateScore());
		postIteration();
	}
	
	/**
	 * Get the network as an array of doubles.
	 * 
	 * @return The network as an array of doubles.
	 */
	public double[] getArray() {
		return NetworkCODEC
				.networkToArray(NeuralSimulatedAnnealing.this.network);
	}

	/**
	 * @return A copy of the annealing array.
	 */
	public double[] getArrayCopy() {
		return getArray();
	}

	/**
	 * Convert an array of doubles to the current best network.
	 * 
	 * @param array
	 *            An array.
	 */
	public void putArray(final double[] array) {
		NetworkCODEC.arrayToNetwork(array,
				NeuralSimulatedAnnealing.this.network);
	}

	/**
	 * Randomize the weights and thresholds. This function does most of the
	 * work of the class. Each call to this class will randomize the data
	 * according to the current temperature. The higher the temperature the
	 * more randomness.
	 */
	public void randomize() {
		final double[] array = NetworkCODEC
				.networkToArray(NeuralSimulatedAnnealing.this.network);

		for (int i = 0; i < array.length; i++) {
			double add = NeuralSimulatedAnnealing.CUT - Math.random();
			add /= this.anneal.getStartTemperature();
			add *= this.anneal.getTemperature();
			array[i] = array[i] + add;
		}

		NetworkCODEC.arrayToNetwork(array,
				NeuralSimulatedAnnealing.this.network);
	}

	/**
	 * @return The object used to calculate the score.
	 */
	public CalculateScore getCalculateScore() {
		return calculateScore;
	}
	
	

}