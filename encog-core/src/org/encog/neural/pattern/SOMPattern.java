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
package org.encog.neural.pattern;

import org.encog.neural.activation.ActivationFunction;
import org.encog.neural.activation.ActivationLinear;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SOMPattern  implements NeuralNetworkPattern {
	
	private int inputNeurons;
	private int outputNeurons;
	
	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());

	@Override
	public void addHiddenLayer(int count) {
		String str = "A SOM network does not have hidden layers.";
		if( logger.isErrorEnabled())
		{
			logger.error(str);
		}
		throw new PatternError(str);
		
	}

	@Override
	public void setActivationFunction(ActivationFunction activation) {
		String str = "A SOM network can't define an activation function.";
		if( logger.isErrorEnabled())
		{
			logger.error(str);
		}
		throw new PatternError(str);
		
	}

	@Override
	public void setInputNeurons(int count) {
		this.inputNeurons = count;
		
	}

	@Override
	public void setOutputNeurons(int count) {
		this.outputNeurons = count;		
	}
	
	@Override
	public BasicNetwork generate() {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationLinear(),false,this.inputNeurons));
		network.addLayer(new BasicLayer(new ActivationLinear(),false,this.outputNeurons));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
}
