/*
 * Encog(tm) Core v2.5 
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

package org.encog.neural.networks.layers;

import org.encog.mathutil.BoundMath;
import org.encog.mathutil.randomize.RangeRandomizer;
import org.encog.mathutil.rbf.GaussianFunction;
import org.encog.mathutil.rbf.RadialBasisFunction;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.activation.ActivationLinear;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.persist.Persistor;
import org.encog.persist.persistors.RadialBasisFunctionLayerPersistor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This layer type makes use of several radial basis function to scale the
 * output from this layer. Each RBF can have a different center, peak, and
 * width. Proper selection of these values will greatly impact the success of
 * the layer. Currently, Encog provides no automated way of determining these
 * values. There is one RBF per neuron.
 *
 * Radial basis function layers have neither bias nor a regular activation
 * function. Calling any methods that deal with the activation function or
 * bias values will result in an error.
 *
 * @author jheaton
 *
 */
public class RadialBasisFunctionLayer extends BasicLayer {

	/**
	 * The serial id.
	 */
	private static final long serialVersionUID = 2779781041654829282L;

	/**
	 * The logging object.
	 */
	private static final transient Logger LOGGER =
		LoggerFactory.getLogger(RadialBasisFunctionLayer.class);

	/**
	 * The radial basis functions to use, there should be one for each neuron.
	 */
	private RadialBasisFunction[] radialBasisFunction;

	/**
	 * Default constructor, mainly so the workbench can easily create a default
	 * layer.
	 */
	public RadialBasisFunctionLayer() {
		this(1);
	}

	/**
	 * Construct a radial basis function layer.
	 *
	 * @param neuronCount
	 *            The neuron count.
	 */
	public RadialBasisFunctionLayer(final int neuronCount) {
		super(new ActivationLinear(), false, neuronCount);
		this.radialBasisFunction = new RadialBasisFunction[neuronCount];
	}

	/**
	 * Compute the values before sending output to the next layer.
	 * This function allows the activation functions to be called.
	 * @param pattern The incoming Project.
	 * @return The output from this layer.
	 */
	@Override
	public NeuralData compute(final NeuralData pattern) {

		final NeuralData result = new BasicNeuralData(getNeuronCount());

		for (int i = 0; i < getNeuronCount(); i++) {

			if (this.radialBasisFunction[i] == null) {
				final String str =
			"Error, must define radial functions for each neuron";
				if (RadialBasisFunctionLayer.LOGGER.isErrorEnabled()) {
					RadialBasisFunctionLayer.LOGGER.error(str);
				}
				throw new NeuralNetworkError(str);
			}

			final RadialBasisFunction f = this.radialBasisFunction[i];
			double total = 0;
			for (int j = 0; j < pattern.size(); j++) {
				final double value = f.calculate(pattern.getData(j));
				total += value * value;
			}

			result.setData(i, BoundMath.sqrt(total));

		}

		return result;
	}

	/**
	 * Create a persistor for this layer.
	 *
	 * @return The new persistor.
	 */
	@Override
	public Persistor createPersistor() {
		return new RadialBasisFunctionLayerPersistor();
	}


	/**
	 * @return An array of radial basis functions.
	 */
	public RadialBasisFunction[] getRadialBasisFunction() {
		return this.radialBasisFunction;
	}

	/**
	 * Set the gausian components to random values.
	 * @param min The minimum value for the centers, widths and peaks.
	 * @param max The maximum value for the centers, widths and peaks.
	 */
	public void randomizeGaussianCentersAndWidths(final double min,
			final double max) {
		for (int i = 0; i < getNeuronCount(); i++) {
			this.radialBasisFunction[i] = new GaussianFunction(RangeRandomizer
					.randomize(min, max), RangeRandomizer.randomize(min, max),
					RangeRandomizer.randomize(min, max));
		}
	}

	/**
	 * Set the RBF array used by this layer.
	 * @param newRBF The new RBF array.
	 */
	public void setRadialBasisFunction(final RadialBasisFunction[] newRBF) {
		this.radialBasisFunction = newRBF;

	}

}
