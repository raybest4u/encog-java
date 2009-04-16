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

package org.encog.neural.activation;

import org.encog.neural.NeuralNetworkError;
import org.encog.persist.Persistor;
import org.encog.persist.persistors.ActivationBiPolarPersistor;
import org.encog.persist.persistors.ActivationTANHPersistor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * BiPolar activation function.  This will scale the neural data into the
 * bipolar range.  Greater than zero becomes 1, less than or equal to zero 
 * becomes -1.
 * @author jheaton
 *
 */
public class ActivationBiPolar extends BasicActivationFunction {
	
	/**
	 * The serial id.
	 */
	private static final long serialVersionUID = -7166136514935838114L;
	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());

	public void activationFunction(double[] d) {
		for(int i=0;i<d.length;i++)
		{
			if( d[i]>0 )
				d[i] = 1;
			else
				d[i] = -1;
		}
		
	}

	public void derivativeFunction(double[] d) {
		throw new NeuralNetworkError(
				"Can't use the bipolar activation function "
						+ "where a derivative is required.");
		
	}

	public Object clone()
	{
		return new ActivationBiPolar();
	}
	
	public Persistor createPersistor() {
		return new ActivationBiPolarPersistor();
	}
}
