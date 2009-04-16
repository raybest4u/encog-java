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
package org.encog.neural.networks.synapse;

import org.encog.neural.networks.layers.Layer;
import org.encog.persist.Persistor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An abstract class that implements basic functionality that may be needed
 * by the other synapse classes. Specifically this class handles processing
 * the from and to layer, as well as providing a name and description for the
 * EncogPersistedObject.
 * @author jheaton
 *
 */
public abstract class BasicSynapse implements Synapse {
	
	private Layer fromLayer;
	private Layer toLayer;
	
	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());
	
	public int getFromNeuronCount() {
		return this.fromLayer.getNeuronCount();		
	}
	
	public int getToNeuronCount() {
		return this.toLayer.getNeuronCount();
	}	


	public Layer getFromLayer() {
		return fromLayer;
	}

	public void setFromLayer(Layer fromLayer) {
		this.fromLayer = fromLayer;
	}

	public Layer getToLayer() {
		return toLayer;
	}

	public void setToLayer(Layer toLayer) {
		this.toLayer = toLayer;
	}
	
	public boolean isSelfConnected()
	{
		return this.fromLayer==this.toLayer;
	}
	
	@Override
	public String toString()
	{
		StringBuilder result = new StringBuilder();
		result.append("[");
		result.append(this.getClass().getSimpleName());
		result.append(": from=");
		result.append(this.getFromNeuronCount());
		result.append(",to=");
		result.append(this.getToNeuronCount());
		result.append("]");
		return result.toString();
	}
	
	abstract public Object clone();
	
	public String getName()
	{
		return null;
	}
	
	public String getDescription()
	{
		return null;
	}
	
	public void setName(String n)
	{
		
	}
	
	public void setDescription(String d)
	{
		
	}
	
	public Persistor createPersistor()
	{
		return null;
	}

}
