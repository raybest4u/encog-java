/*
 * Encog(tm) Core v2.5 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2010 Heaton Research, Inc.
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

package org.encog.mathutil.rbf;

import org.encog.engine.network.rbf.RadialBasisFunction;
import org.encog.engine.util.BoundMath;

/**
 * Multi-dimensional Multiquadric function. Do not use this to implement a 1d
 * function, simply use MultiquadricFunction for that.
 *
 */
public class MultiquadricFunction implements RadialBasisFunction {
	/**
	 * The center of the RBF.
	 */
	private double[] center;

	/**
	 * The peak of the RBF.
	 */
	private double peak;

	/**
	 * The width of the RBF.
	 */
	private double width;

	/**
	 * Create centered at zero, width 0, and peak 0.
	 */
	public MultiquadricFunction(int dimensions)
	{
		this.center = new double[dimensions];
		this.peak = 1.0;
		this.width = 1.0;		
	}
	
	/**
	 * Construct a multi-dimension Multiquadric function with the specified
	 * peak, centers and widths.
	 * @param peak The peak for all dimensions.
	 * @param center The centers for each dimension.
	 * @param width The widths for each dimension.
	 */
	public MultiquadricFunction(final double peak, final double[] center,
			final double width) {
		this.center = center;
		this.peak = peak;
		this.width = width;
	}

	/**
	 * Construct a Multiquadric function with the specified number of
	 * dimensions. The peak, center and widths are all the same.
	 * @param dimensions The number of dimensions.
	 * @param peak The peak used for all dimensions.
	 * @param center The center used for all dimensions.
	 * @param width The widths used for all dimensions.
	 */
	public MultiquadricFunction(final int dimensions, final double peak,
			final double center, final double width) {
		this.peak = peak;
		this.center = new double[dimensions];
		this.width = width;
	}

	/**
	 * {@inheritDoc}
	 */
	public double calculate(final double[] x) {
		double value = 0;

		for (int i = 0; i < this.center.length; i++) {
			value += Math.pow(x[i] - this.center[i], 2)
					+ (this.width * this.width);
		}
		return this.peak * BoundMath.sqrt(value);
	}

	/**
	 * {@inheritDoc}
	 */
	public double[] getCenter() {
		return this.center;
	}

	/**
	 * {@inheritDoc}
	 */
	public double getCenter(final int dimension) {
		return this.center[dimension];
	}

	/**
	 * {@inheritDoc}
	 */
	public int getDimensions() {
		return this.center.length;
	}

	/**
	 * {@inheritDoc}
	 */
	public double getPeak() {
		return this.peak;
	}

	/**
	 * {@inheritDoc}
	 */
	public double getWidth() {
		return this.width;
	}

	/**
	 * {@inheritDoc}
	 */
	public void setCenter(final double[] center) {
		this.center = center;
	}

	/**
	 * {@inheritDoc}
	 */
	public void setPeak(final double peak) {
		this.peak = peak;
	}

	/**
	 * {@inheritDoc}
	 */
	public void setWidth(final double w) {
		this.width = w;
	}
	
	/**
	 * @return The centers.
	 */
	@Override
	public double[] getCenters() {
		return this.center;
	}


}
