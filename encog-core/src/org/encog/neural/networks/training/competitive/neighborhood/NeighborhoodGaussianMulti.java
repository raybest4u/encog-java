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

package org.encog.neural.networks.training.competitive.neighborhood;

import org.encog.mathutil.rbf.RBFEnum;


/**
 * Implements a multi-dimensional gaussian neighborhood function.  DO not
 * use this for a 1D gaussian, just use the NeighborhoodGaussian for that.
 *
 */
public class NeighborhoodGaussianMulti extends NeighborhoodRBF {

	/**
	 * Construct a 2d neighborhood function based on the sizes for the
	 * x and y dimensions.
	 * @param x The size of the x-dimension.
	 * @param y The size of the y-dimension.
	 */
	public NeighborhoodGaussianMulti( 
			final int x, final int y) {
		super(RBFEnum.Gaussian,x,y);
	}

}
