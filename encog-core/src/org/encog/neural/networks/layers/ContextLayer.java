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

package org.encog.neural.networks.layers;

import org.encog.neural.activation.ActivationFunction;
import org.encog.neural.activation.ActivationTANH;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.persist.Persistor;
import org.encog.persist.persistors.BasicLayerPersistor;
import org.encog.persist.persistors.ContextLayerPersistor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implements a context layer.  A context layer is used to implement
 * a simple recurrent neural network, such as an Elman or Jordan neural
 * network.  The context layer has a short-term memory.  The context layer 
 * accept input, and provide the same data as output on the next cycle.
 * This continues, and the context layer's output "one step" out of sync
 * with the input.
 * @author jheaton
 *
 */
public class ContextLayer extends BasicLayer {

	/**
	 * The serial id.
	 */
	private static final long serialVersionUID = -5588659547177460637L;

	private NeuralData context;
	
	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());
	
	public ContextLayer(ActivationFunction thresholdFunction, boolean hasThreshold, int neuronCount) {
		super(thresholdFunction, hasThreshold, neuronCount);
		context = new BasicNeuralData(neuronCount);
	}
	
	public ContextLayer(int neuronCount) {
		this(new ActivationTANH(), true, neuronCount);
	}
	
	/**
	 * Default constructor, mainly so the workbench can easily create a default layer.
	 */
	public ContextLayer()
	{
		this(1);
	}
	
	public NeuralData recur() {
		return this.context;
	}
	
	@Override
	public void process(NeuralData pattern) {
		for(int i = 0; i<pattern.size();i++)
		{
			this.context.setData(i,pattern.getData(i));
		}
		
		if( logger.isDebugEnabled() ) {
			logger.debug("Updated ContextLayer to {}", pattern);
		}		
	}
	
	/**
	 * Create a persistor for this layer.
	 * @return The new persistor.
	 */
	public Persistor createPersistor() {
		return new ContextLayerPersistor();
	}
	

}
