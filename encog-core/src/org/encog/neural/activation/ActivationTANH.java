/*
 * Encog(tm) Core v2.4
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

/**
 * The hyperbolic tangent activation function takes the curved shape of the
 * hyperbolic tangent. This activation function produces both positive and
 * negative output. Use this activation function if both negative and positive
 * output is desired.
 * 
 */
public class ActivationTANH extends BasicActivationFunction  implements SlopeActivationFunction {

	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = 9121998892720207643L;

	public static final String[] PARAM_NAMES = {
	"slope" };
	
	/**
	 * The parameter for the slope.
	 */
	public static final int PARAM_SLOPE = 0;
	
	
	public ActivationTANH()
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
				ActivationFunctions.ACTIVATION_TANH, 
				d, 
				this.params,
				0,
				d.length,
				0);

	}

	/**
	 * @return The object cloned;
	 */
	@Override
	public Object clone() {
		return new ActivationTANH();
	}

	/**
	 * Calculate the derivative of the activation. It is assumed that the value
	 * d, which is passed to this method, was the output from this activation.
	 * This prevents this method from having to recalculate the activation, just
	 * to recalculate the derivative.
	 * 
	 * The array is modified according derivative of the activation function
	 * being used. See the class description for more specific information on
	 * this type of activation function. Propagation training requires the
	 * derivative. Some activation functions do not support a derivative and
	 * will throw an error.
	 * 
	 * @param d
	 *            The input array to the activation function.
	 */
	public double derivativeFunction(final double d) {
		return ActivationFunctions.calculateActivationDerivative(
				ActivationFunctions.ACTIVATION_TANH, 
				d, 
				this.params,
				0);
	}

	/**
	 * @return Return true, TANH has a derivative.
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
	
	/**
	 * @return The Encog Engine ID for this activation type, or -1 if not
	 *         defined by the Encog engine.
	 */
	@Override
	public int getEngineID() {
		// TODO Auto-generated method stub
		return ActivationFunctions.ACTIVATION_TANH;
	}

}
