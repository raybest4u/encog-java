/*
 * Encog Artificial Intelligence Framework v2.x
 * Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2009, Heaton Research Inc., and individual contributors.
 * See the copyright.txt in the distribution for a full listing of 
 * individual contributors.
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
 */

package org.encog.neural.networks.training.propagation.resilient;

import org.encog.matrix.Matrix;
import org.encog.neural.networks.NeuralOutputHolder;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.training.propagation.CalculatePartialDerivative;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.PropagationLevel;
import org.encog.neural.networks.training.propagation.PropagationMethod;
import org.encog.neural.networks.training.propagation.PropagationSynapse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implements the specifics of the resilient propagation training algorithm.
 * 
 * @author jheaton
 * 
 */
public class ResilientPropagationMethod implements PropagationMethod {

	/**
	 * The logging object.
	 */
	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * The propagation class that this method is used with.
	 */
	private ResilientPropagation propagation;
	
	/**
	 * Utility class to calculate the partial derivative.
	 */
	private final CalculatePartialDerivative pderv 
		= new CalculatePartialDerivative();

	/**
	 * Calculate the error between these two levels.
	 * 
	 * @param output
	 *            The output to the "to level".
	 * @param fromLevel
	 *            The from level.
	 * @param toLevel
	 *            The target level.
	 */
	public void calculateError(final NeuralOutputHolder output,
			final PropagationLevel fromLevel, final PropagationLevel toLevel) {

		this.pderv.calculateError(output, fromLevel, toLevel);

	}

	/**
	 * Init with the specified propagation object.
	 * 
	 * @param propagation
	 *            The propagation object that this method will be used with.
	 */
	public void init(final Propagation propagation) {
		this.propagation = (ResilientPropagation) propagation;

	}

	/**
	 * Modify the weight matrix and thresholds based on the last call to
	 * calcError.
	 */
	public void learn() {
		if (this.logger.isDebugEnabled()) {
			this.logger.debug("Backpropagation learning pass");
		}

		for (final PropagationLevel level : this.propagation.getLevels()) {
			learnLevel(level);
		}
	}

	/**
	 * Apply the learning to the specified level.
	 * @param level The level that is to learn.
	 */
	private void learnLevel(final PropagationLevel level) {
		// teach the synapses
		for (final PropagationSynapse synapse : level.getOutgoing()) {
			learnSynapse(synapse);
		}

		// teach the threshold
		for (final Layer layer : level.getLayers()) {
			if (layer.hasThreshold()) {
				for (int i = 0; i < layer.getNeuronCount(); i++) {

					// multiply the current and previous gradient, and take the
					// sign. We want to see if the gradient has changed its
					// sign.
					final int change = sign(level.getThresholdGradient(i)
							* level.getLastThresholdGradent(i));
					double weightChange = 0;

					// if the gradient has retained its sign, then we increase
					// the delta so that it will converge faster
					if (change > 0) {
						double delta = level.getThresholdDelta(i)
								* ResilientPropagation.POSITIVE_ETA;
						delta = Math.min(delta, this.propagation.getMaxStep());
						weightChange = sign(level.getThresholdGradient(i))
								* delta;
						level.setThresholdDelta(i, delta);
						level.setLastThresholdGradient(i, level
								.getThresholdGradient(i));
					} else if (change < 0) {
						// if change<0, then the sign has changed, and the last
						// delta was too big
						double delta = level.getThresholdDelta(i)
								* ResilientPropagation.NEGATIVE_ETA;
						delta = Math.max(delta, ResilientPropagation.DELTA_MIN);
						level.setThresholdDelta(i, delta);
						// set the previous gradient to zero so that there will
						// be no adjustment the next iteration
						level.setLastThresholdGradient(i, 0);
					} else if (change == 0) {
						// if change==0 then there is no change to the delta
						final double delta = level.getThresholdDelta(i);
						weightChange = sign(level.getThresholdGradient(i))
								* delta;
						level.setLastThresholdGradient(i, level
								.getThresholdGradient(i));
					}

					// apply the weight change, if any
					layer.setThreshold(i, layer.getThreshold(i) + weightChange);

					level.setThresholdGradient(i, 0.0);
				}
			}
		}

	}

	/**
	 * Learn from the last error calculation.
	 *
	 * @param synapse The synapse to teach.
	 */
	private void learnSynapse(final PropagationSynapse synapse) {

		final Matrix matrix = synapse.getSynapse().getMatrix();

		for (int row = 0; row < matrix.getRows(); row++) {
			for (int col = 0; col < matrix.getCols(); col++) {
				// multiply the current and previous gradient, and take the
				// sign. We want to see if the gradient has changed its sign.
				final int change = sign(synapse.getAccMatrixGradients().get(
						row, col)
						* synapse.getLastMatrixGradients().get(row, col));
				double weightChange = 0;

				// if the gradient has retained its sign, then we increase the
				// delta so that it will converge faster
				if (change > 0) {
					double delta = synapse.getDeltas().get(row, col)
							* ResilientPropagation.POSITIVE_ETA;
					delta = Math.min(delta, this.propagation.getMaxStep());
					weightChange = sign(synapse.getAccMatrixGradients().get(
							row, col))
							* delta;
					synapse.getDeltas().set(row, col, delta);
					synapse.getLastMatrixGradients().set(row, col,
							synapse.getAccMatrixGradients().get(row, col));
				} else if (change < 0) {
					// if change<0, then the sign has changed, and the last 
					// delta was too big
					double delta = synapse.getDeltas().get(row, col)
							* ResilientPropagation.NEGATIVE_ETA;
					delta = Math.max(delta, ResilientPropagation.DELTA_MIN);
					synapse.getDeltas().set(row, col, delta);
					// set the previous gradent to zero so that there will be no
					// adjustment the next iteration
					synapse.getLastMatrixGradients().set(row, col, 0);
				} else if (change == 0) {
					// if change==0 then there is no change to the delta
					final double delta = synapse.getDeltas().get(row, col);
					weightChange = sign(synapse.getAccMatrixGradients().get(
							row, col))
							* delta;
					synapse.getLastMatrixGradients().set(row, col,
							synapse.getAccMatrixGradients().get(row, col));
				}

				// apply the weight change, if any
				matrix.set(row, col, synapse.getSynapse().getMatrix().get(row,
						col)
						+ weightChange);

			}
		}

		// clear out the gradient accumulator for the next iteration
		synapse.getAccMatrixGradients().clear();
	}

	/**
	 * Determine the sign of the value.  
	 * @param value The value to check.
	 * @return -1 if less than zero, 1 if greater, or 0 if zero.
	 */
	private int sign(final double value) {
		if (Math.abs(value) < this.propagation.getZeroTolerance()) {
			return 0;
		} else if (value > 0) {
			return 1;
		} else {
			return -1;
		}
	}

}
