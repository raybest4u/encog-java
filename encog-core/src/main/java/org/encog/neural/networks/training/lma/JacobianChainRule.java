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
package org.encog.neural.networks.training.lma;

import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.activation.ActivationFunction;
import org.encog.neural.networks.BasicNetwork;

/**
 * Calculate the Jacobian using the chain rule.
 * ----------------------------------------------------
 * 
 * This implementation of the Levenberg Marquardt algorithm is based heavily on code
 * published in an article by Cesar Roberto de Souza.  The original article can be
 * found here:
 * 
 * http://crsouza.blogspot.com/2009/11/neural-network-learning-by-levenberg_18.html
 * 
 * Portions of this class are under the following copyright/license.
 * Copyright 2009 by Cesar Roberto de Souza, Released under the LGPL.
 */
public class JacobianChainRule implements ComputeJacobian {
	/**
	 * The network that is to be trained.
	 */
	private final BasicNetwork network;

	/**
	 * THe training set to use. Must be indexable.
	 */
	private final MLDataSet indexableTraining;

	/**
	 * The number of training set elements.
	 */
	private final int inputLength;

	/**
	 * The number of weights and bias values in the neural network.
	 */
	private final int parameterSize;

	/**
	 * The Jacobian matrix that was calculated.
	 */
	private final double[][] jacobian;

	/**
	 * The current row in the Jacobian matrix.
	 */
	private int jacobianRow;

	/**
	 * The current column in the Jacobian matrix.
	 */
	private int jacobianCol;

	/**
	 * Used to read the training data.
	 */
	private final MLDataPair pair;

	/**
	 * The errors for each row in the Jacobian.
	 */
	private final double[] rowErrors;

	/**
	 * The current error.
	 */
	private double error;

	/**
	 * Construct the chain rule calculation.
	 * 
	 * @param network
	 *            The network to use.
	 * @param indexableTraining
	 *            The training set to use.
	 */
	public JacobianChainRule(final BasicNetwork network,
			final MLDataSet indexableTraining) {
		this.indexableTraining = indexableTraining;
		this.network = network;
		this.parameterSize = network.getStructure().calculateSize();
		this.inputLength = (int) this.indexableTraining.getRecordCount();
		this.jacobian = new double[this.inputLength][this.parameterSize];
		this.rowErrors = new double[this.inputLength];

		final BasicMLData input = new BasicMLData(
				this.indexableTraining.getInputSize());
		final BasicMLData ideal = new BasicMLData(
				this.indexableTraining.getIdealSize());
		this.pair = new BasicMLDataPair(input, ideal);
	}

	/**
	 * Calculate the derivative.
	 * 
	 * @param a
	 *            The activation function.
	 * @param d
	 *            The value to calculate for.
	 * @return The derivative.
	 */
	private double calcDerivative(final ActivationFunction a, final double d) {

		return a.derivativeFunction(d);
	}

	/**
	 * Calculate the derivative.
	 * 
	 * @param a
	 *            The activation function.
	 * @param d
	 *            The value to calculate for.
	 * @return The derivative.
	 */
	private double calcDerivative2(final ActivationFunction a, final double d) {
		final double[] temp = new double[1];
		temp[0] = d;
		a.activationFunction(temp,0,temp.length);
		temp[0] = a.derivativeFunction(temp[0]);
		return temp[0];
	}

	/**
	 * Calculate the Jacobian matrix.
	 * 
	 * @param weights
	 *            The weights for the neural network.
	 * @return The sum squared of the weights.
	 */
	public double calculate(final double[] weights) {
		double result = 0.0;

		for (int i = 0; i < this.inputLength; i++) {
			this.jacobianRow = i;
			this.jacobianCol = 0;

			this.indexableTraining.getRecord(i, this.pair);

			final double e = calculateDerivatives(this.pair);
			this.rowErrors[i] = e;
			result += e * e;

		}

		return result / 2.0;
	}

	/**
	 * Calculate the derivatives for this training set element.
	 * 
	 * @param pair
	 *            The training set element.
	 * @return The sum squared of errors.
	 */
	private double calculateDerivatives(final MLDataPair pair) {
		// error values
		double e = 0.0;
		double sum = 0.0;

		this.network.compute(pair.getInput());

		int fromLayer = this.network.getLayerCount()-2;
		int toLayer = this.network.getLayerCount()-1;
		int fromNeuronCount = this.network.getLayerTotalNeuronCount(fromLayer);
		int toNeuronCount = this.network.getLayerNeuronCount(toLayer);

		double output = this.network.getStructure().getFlat().getLayerOutput()[0];
		e = pair.getIdeal().getData(0) - output;

		for (int i = 0; i < fromNeuronCount; i++) {
			final double lastOutput = network.getLayerOutput(fromLayer,i);

			this.jacobian[this.jacobianRow][this.jacobianCol++] 
			    = calcDerivative(
					this.network.getActivation(toLayer), output)
					* lastOutput;
		}

		while (fromLayer >0 ) {

			fromLayer--;
			toLayer--;
			fromNeuronCount = this.network.getLayerTotalNeuronCount(fromLayer);
			toNeuronCount = this.network.getLayerNeuronCount(toLayer);
			
			// this.network.getLayerOutput(fromLayer, neuronNumber) holder.getResult().get(lastSynapse);

			// for each neuron in the input layer
			for (int neuron = 0; neuron 
			  < toNeuronCount; neuron++) {
				output = this.network.getLayerOutput(toLayer, neuron);
				
				ActivationFunction function = network.getActivation(toLayer);
				
				final double w = network.getWeight(toLayer, neuron, 0);
				final double val = calcDerivative(function, output)
						* calcDerivative2(function, sum) * w;

				// for each weight of the input neuron
				for (int i = 0; i < fromNeuronCount; i++) {
					sum = 0.0;
					// for each neuron in the next layer
					for (int j = 0; j < toNeuronCount; j++) {
						// for each weight of the next neuron
						for (int k = 0; k 
						  < fromNeuronCount; k++) {
							sum += network.getWeight(fromLayer, k, j) * output;
						}
					}

					this.jacobian[this.jacobianRow][this.jacobianCol++] = val
							* this.network.getLayerOutput(fromLayer, i);
				}
			
			}
		}

		// return error
		return e;
	}

	/**
	 * @return The sum squared errors.
	 */
	public double getError() {
		return this.error;
	}

	/**
	 * @return The Jacobian matrix.
	 */
	public double[][] getJacobian() {
		return this.jacobian;
	}

	/**
	 * @return The errors for each row of the Jacobian.
	 */
	public double[] getRowErrors() {
		return this.rowErrors;
	}
}
