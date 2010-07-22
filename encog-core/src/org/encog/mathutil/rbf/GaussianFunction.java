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

package org.encog.mathutil.rbf;

import java.io.Serializable;

import org.encog.engine.util.BoundMath;

/**
 * Implements a radial function based on the gaussian function.
 * 
 * @author jheaton
 * 
 */
public class GaussianFunction implements RadialBasisFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 548203092442332198L;

	/**
	 * The center of the RBF.
	 */
	private final double center;

	/**
	 * The peak of the RBF.
	 */
	private final double peak;

	/**
	 * The width of the RBF.
	 */
	private double width;

	/**
	 * Construct a Gaussian RBF with the specified center, peak and width.
	 * 
	 * @param center
	 *            The center.
	 * @param peak
	 *            The peak.
	 * @param width
	 *            The width.
	 */
	public GaussianFunction(final double center, final double peak,
			final double width) {
		this.center = center;
		this.peak = peak;
		this.width = width;
	}

	/**
	 * Calculate the value of the Gaussian function for the specified value.
	 * 
	 * @param x
	 *            The value to calculate the Gaussian function for.
	 * @return The return value for the Gaussian function.
	 */
	public double calculate(final double x) {
		return this.peak
				* BoundMath.exp(-Math.pow(x - this.center, 2)
						/ (2.0 * this.width * this.width));
	}

	/**
	 * Calculate the value of the derivative of the Gaussian function for the
	 * specified value.
	 * 
	 * @param x
	 *            The value to calculate the derivative Gaussian function for.
	 * @return The return value for the derivative of the Gaussian function.
	 */
	public double calculateDerivative(final double x) {
		return Math.exp(-0.5 * this.width * this.width * x * x) * this.peak
				* this.width * this.width
				* (this.width * this.width * x * x - 1);
	}

	/**
	 * @return The center of the RBF.
	 */
	public double getCenter() {
		return this.center;
	}

	/**
	 * @return The peak of the RBF.
	 */
	public double getPeak() {
		return this.peak;
	}

	/**
	 * @return The width of the RBF.
	 */
	public double getWidth() {
		return this.width;
	}

	/**
	 * Set the width of the function.
	 * 
	 * @param radius
	 *            The width.
	 */
	public void setWidth(final double radius) {
		this.width = radius;
	}

}
