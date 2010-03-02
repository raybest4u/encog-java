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

package org.encog.normalize.input;

import org.encog.neural.data.NeuralDataSet;
import org.encog.persist.annotations.EGUnsupported;

/**
 * An input field based on an Encog NeuralDataSet.
 *
 */
@EGUnsupported
public class InputFieldNeuralDataSet extends BasicInputField {

	/**
	 * The data set.
	 */
	private final NeuralDataSet data;
	
	/**
	 * The input or ideal index.  This treats the input and ideal as one
	 * long array, concatenated together.
	 */
	private final int offset;

	/**
	 * Construct a input field based on a NeuralDataSet.
	 * @param usedForNetworkInput Is this field used for neural input.
	 * @param data The data set to use.
	 * @param offset The input or ideal index to use. This treats the input 
	 * and ideal as one long array, concatenated together.
	 */
	public InputFieldNeuralDataSet(final boolean usedForNetworkInput,
			final NeuralDataSet data, final int offset) {
		this.data = data;
		this.offset = offset;
		setUsedForNetworkInput(usedForNetworkInput);
	}

	/**
	 * @return The neural data set to read.
	 */
	public NeuralDataSet getNeuralDataSet() {
		return this.data;
	}

	/**
	 * @return The field to be accessed. This treats the input and 
	 * ideal as one long array, concatenated together.
	 */
	public int getOffset() {
		return this.offset;
	}

}
