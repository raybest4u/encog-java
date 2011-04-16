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
package org.encog.neural.pnn;

import org.encog.mathutil.EncogMath;
import org.encog.ml.MLRegression;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataPair;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.neural.data.basic.BasicNeuralDataSet;

/**
 * This class implements either a:
 * 
 * Probabilistic Neural Network (PNN) 
 * 
 * General Regression Neural Network (GRNN)
 * 
 * To use a PNN specify an output mode of classification, to make use of a GRNN
 * specify either an output mode of regression or un-supervised autoassociation.
 * 
 * The PNN/GRNN networks are potentially very useful. They share some
 * similarities with RBF-neural networks and also the Support Vector Machine
 * (SVM). These network types directly support the use of classification.
 * 
 * The following book was very helpful in implementing PNN/GRNN's in Encog.
 * 
 * Advanced Algorithms for Neural Networks: A C++ Sourcebook 
 * 
 * by Timothy Masters, PhD (http://www.timothymasters.info/)
 * John Wiley & Sons Inc (Computers); April 3, 1995, ISBN: 0471105880
 */
public class BasicPNN extends AbstractPNN implements MLRegression {

	/**
	 * The sigma's specify the widths of each kernel used.
	 */
	private double[] sigma;

	/**
	 * The training samples that form the memory of this network.
	 */
	private BasicNeuralDataSet samples;

	/**
	 * Used for classification, the number of cases in each class.
	 */
	private int[] countPer;

	/**
	 * The prior probability weights. 
	 */
	private double[] priors;

	/**
	 * Construct a BasicPNN network.
	 * @param kernel The kernel to use.
	 * @param outmodel The output model for this network.
	 * @param inputCount The number of inputs in this network.
	 * @param outputCount The number of outputs in this network.
	 */
	public BasicPNN(final PNNKernelType kernel, final PNNOutputMode outmodel,
			final int inputCount, final int outputCount) {
		super(kernel, outmodel, inputCount, outputCount);

		this.setSeparateClass(false);

		this.sigma = new double[inputCount];
	}

	/**
	 * Compute the output from this network.
	 * @param input The input to the network.
	 * @return The output from the network.
	 */
	@Override
	public NeuralData compute(final NeuralData input) {

		final double[] out = new double[getOutputCount()];

		double psum = 0.0;

		int r = -1;
		for (NeuralDataPair pair : this.samples) {
			r++;

			if (r == getExclude()) {
				continue;
			}

			double dist = 0.0;
			for (int i = 0; i < getInputCount(); i++) {
				double diff = input.getData(i) - pair.getInput().getData(i);
				diff /= this.sigma[i];
				dist += diff * diff;
			}

			if (getKernel() == PNNKernelType.Gaussian) {
				dist = Math.exp(-dist);
			} else if (getKernel() == PNNKernelType.Reciprocal) {
				dist = 1.0 / (1.0 + dist);
			}

			if (dist < 1.e-40) {
				dist = 1.e-40;
			}

			if (getOutputMode() == PNNOutputMode.Classification) {
				int pop = (int) pair.getIdeal().getData(0);
				out[pop] += dist;
			} else if (getOutputMode() == PNNOutputMode.Unsupervised) {
				for (int i = 0; i < getInputCount(); i++) {
					out[i] += dist * pair.getInput().getData(i);
				}
				psum += dist;
			} else if (getOutputMode() == PNNOutputMode.Regression) {

				for (int i = 0; i < getOutputCount(); i++) {
					out[i] += dist * pair.getIdeal().getData(i);
				}

				psum += dist;
			}
		}

		if (getOutputMode() == PNNOutputMode.Classification) {
			psum = 0.0;
			for (int i = 0; i < getOutputCount(); i++) {
				if (this.priors[i] >= 0.0) {
					out[i] *= this.priors[i] / this.countPer[i];
				}
				psum += out[i];
			}

			if (psum < 1.e-40) {
				psum = 1.e-40;
			}

			for (int i = 0; i < getOutputCount(); i++) {
				out[i] /= psum;
			}

			final NeuralData result = new BasicNeuralData(1);
			result.setData(0, EncogMath.maxIndex(out));
			return result;
		}

		else if (getOutputMode() == PNNOutputMode.Unsupervised) {
			for (int i = 0; i < getInputCount(); i++) {
				out[i] /= psum;
			}
		}

		else if (getOutputMode() == PNNOutputMode.Regression) {
			for (int i = 0; i < getOutputCount(); i++) {
				out[i] /= psum;
			}
		}

		return new BasicNeuralData(out);
	}

	/**
	 * @return the sigma
	 */
	public double[] getSigma() {
		return sigma;
	}

	/**
	 * @return the samples
	 */
	public BasicNeuralDataSet getSamples() {
		return samples;
	}

	/**
	 * @return the countPer
	 */
	public int[] getCountPer() {
		return countPer;
	}

	/**
	 * @return the priors
	 */
	public double[] getPriors() {
		return priors;
	}

	/**
	 * @param samples
	 *            the samples to set
	 */
	public void setSamples(BasicNeuralDataSet samples) {
		this.samples = samples;

		// update counts per
		if (getOutputMode() == PNNOutputMode.Classification) {

			this.countPer = new int[getOutputCount()];
			this.priors = new double[getOutputCount()];
			
			for(NeuralDataPair pair: samples) {
				int i = (int)pair.getIdeal().getData(0);
				this.countPer[i]++;
			}
			
			for(int i=0;i<this.priors.length;i++) {
				this.priors[i] = -1;
			}

		}
	}

	@Override
	public void updateProperties() {
		// unneeded
		
	}
}
