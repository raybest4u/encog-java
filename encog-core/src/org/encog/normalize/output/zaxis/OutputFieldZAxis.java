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
package org.encog.normalize.output.zaxis;

import org.encog.normalize.NormalizationError;
import org.encog.normalize.input.InputField;
import org.encog.normalize.output.OutputFieldGroup;
import org.encog.normalize.output.OutputFieldGrouped;

public class OutputFieldZAxis extends OutputFieldGrouped {

	public OutputFieldZAxis(final OutputFieldGroup group, final InputField field) {
		super(group, field);
		if (!(group instanceof ZAxisGroup)) {
			throw new NormalizationError(
					"Must use ZAxisGroup with OutputFieldZAxis.");
		}
	}

	public double calculate(int subfield) {
		return (getSourceField().getCurrentValue() * ((ZAxisGroup) getGroup())
				.getMultiplier());
	}
	
	public int getSubfieldCount()
	{
		return 1;
	}
	
	/**
	 * Not needed for this sort of output field.
	 */
	public void rowInit()
	{		
	}


}
