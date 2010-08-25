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

package org.encog.neural.activation;

import org.encog.engine.network.flat.ActivationFunctions;
import org.encog.persist.Persistor;

/**
 * The Linear layer is really not an activation function at all. The input is
 * simply passed on, unmodified, to the output. This activation function is
 * primarily theoretical and of little actual use. Usually an activation
 * function that scales between 0 and 1 or -1 and 1 should be used.
 */
public class ActivationLinear extends BasicActivationFunction implements SlopeActivationFunction {

	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = -5356580554235104944L;

	public static final String[] PARAM_NAMES = {
	"slope" };	

	
	/**
	 * The parameter for the slope.
	 */
	public static final int PARAM_SLOPE = 0;
	
	
	public ActivationLinear()
	{
		this.params = new double[1];
		this.params[PARAM_SLOPE] = 1;
	}
	
	/**
	 * Implements the activation function. The array is modified according to
	 * the activation function being used. See the class description for more
	 * specific information on this type of activation function.
	 * 
	 * @param d
	 *            The input array to the activation function.
	 */
	public void activationFunction(final double[] d) {
		ActivationFunctions.calculateActivation(
				ActivationFunctions.ACTIVATION_LINEAR, 
				d, 
				this.params,
				0,
				d.length,
				0);
	}

	/**
	 * @return The object cloned.
	 */
	@Override
	public Object clone() {
		return new ActivationLinear();
	}

	/**
	 * Implements the activation function derivative. The array is modified
	 * according derivative of the activation function being used. See the class
	 * description for more specific information on this type of activation
	 * function. Propagation training requires the derivative. Some activation
	 * functions do not support a derivative and will throw an error.
	 * 
	 * @param d
	 *            The input array to the activation function.
	 */
	public double derivativeFunction(final double x) {
		return ActivationFunctions.calculateActivationDerivative(
				ActivationFunctions.ACTIVATION_LINEAR, 
				x, 
				this.params,
				0);
	}

	/**
	 * @return Return true, linear has a 1 derivative.
	 */
	public boolean hasDerivative() {
		return true;
	}
	
	/**
	 * Get the slope of the activation function.
	 */
	public double getSlope()
	{
		return this.params[PARAM_SLOPE];
	}
	
	/**
	 * @return The paramater names for this activation function.
	 */
	@Override
	public String[] getParamNames() {
		return PARAM_NAMES;
	}
}
