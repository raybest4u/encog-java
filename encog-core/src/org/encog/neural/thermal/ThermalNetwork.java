/*
 * Encog(tm) Core v3.0 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package org.encog.neural.thermal;

import org.encog.engine.util.EngineArray;
import org.encog.ml.BasicML;
import org.encog.ml.MLAutoAssocation;
import org.encog.ml.MLMethod;
import org.encog.ml.MLResettable;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.data.bipolar.BiPolarNeuralData;
import org.encog.persist.BasicPersistedObject;

public abstract class ThermalNetwork extends BasicML implements MLMethod, MLAutoAssocation, MLResettable {

	/**
	 * The current state of the thermal network.
	 */
	private BiPolarNeuralData currentState;
	
	private double[] weights;
	
	private int neuronCount;
	
	public ThermalNetwork()
	{
		
	}
	
	public ThermalNetwork(int neuronCount) {
		this.neuronCount = neuronCount;
		this.weights = new double[neuronCount*neuronCount];
		this.currentState = new BiPolarNeuralData(neuronCount);		
	}

	/**
	 * @return Calculate the current energy for the network. The network will
	 *         seek to lower this value.
	 */
	public double calculateEnergy() {
		double tempE = 0;
		final int neuronCount = getNeuronCount();

		for (int i = 0; i < neuronCount; i++) {
			for (int j = 0; j < neuronCount; j++) {
				if (i != j) {
					tempE += this.getWeight(i, j)
							* this.currentState.getData(i)
							* this.currentState.getData(j);
				}
			}
		}
		return -1 * tempE / 2;

	}

	/**
	 * Clear any connection weights.
	 */
	public void clear() {
		EngineArray.fill(this.weights, 0);
	}

	/**
	 * @return The current state of the network.
	 */
	public BiPolarNeuralData getCurrentState() {
		return this.currentState;
	}

	/**
	 * @return Get the neuron count for the network.
	 */
	public int getNeuronCount() {
		return this.neuronCount;
	}

	/**
	 * @param state
	 *            The current state for the network.
	 */
	public void setCurrentState(final BiPolarNeuralData state) {
		for (int i = 0; i < state.size(); i++) {
			this.currentState.setData(i, state.getData(i));
		}
	}
	
	public double[] getWeights()
	{
		return this.weights;
	}
	
	public double getWeight(int fromNeuron, int toNeuron)
	{
		int index = (toNeuron*neuronCount) + fromNeuron;
		return weights[index];
	}
	
	public void setWeight(int fromNeuron, int toNeuron, double value)
	{
		int index = (toNeuron*neuronCount) + fromNeuron;
		weights[index] = value;
	}
	
	public void addWeight(int fromNeuron, int toNeuron, double value)
	{
		int index = (toNeuron*neuronCount) + fromNeuron;
		if( index>=weights.length ) {
			throw new NeuralNetworkError("Out of range: fromNeuron:" 
					+ fromNeuron 
					+ ", toNeuron: " 
					+ toNeuron);
		}
		weights[index] += value;
	}
	
	public void init(int neuronCount, double[] weights, double[] output)
	{
		if( neuronCount!=output.length )
		{
			throw new NeuralNetworkError("Neuron count(" + neuronCount + ") must match output count("+output.length+").");
		}
		
		if( (neuronCount*neuronCount)!=weights.length )
		{
			throw new NeuralNetworkError("Weight count(" + weights.length + ") must be the square of the neuron count("+neuronCount+").");
		}
		
		this.neuronCount = neuronCount;
		this.weights = weights;
		this.currentState = new BiPolarNeuralData(neuronCount);
		this.currentState.setData(output);
	}
	
	public void reset(int seed) {
		this.getCurrentState().clear();
		EngineArray.fill(this.weights, 0.0);
	}
	
	public void reset() {
		reset(0);
	}
	
	
}
