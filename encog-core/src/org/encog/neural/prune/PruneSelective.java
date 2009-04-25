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
package org.encog.neural.prune;

import java.util.Collection;

import org.encog.matrix.Matrix;
import org.encog.matrix.MatrixMath;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.synapse.Synapse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class PruneSelective {
	
	private BasicNetwork network;
	
	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());
	
	public PruneSelective(BasicNetwork network)
	{
		this.network = network;
	}
	
	public void changeNeuronCount(Layer layer, int neuronCount)
	{
		// is there anything to do?
		if( neuronCount==layer.getNeuronCount())
			return;
		
		if( neuronCount>layer.getNeuronCount() )
			increaseNeuronCount(layer,neuronCount);
		else
			decreaseNeuronCount(layer,neuronCount);
	}
	
	private void increaseNeuronCount(Layer layer,int neuronCount)
	{
		// adjust the threshold
		double[] newThreshold = new double[neuronCount];
		for(int i=0;i<layer.getNeuronCount();i++)
		{
			newThreshold[i] = layer.getThreshold(i);
		}
		
		layer.setThreshold(newThreshold);
		
		// adjust the outbound weight matrixes
		for(Synapse synapse: layer.getNext())
		{
			Matrix newMatrix = new Matrix(neuronCount,synapse.getToNeuronCount());
			// copy existing matrix to new matrix
			for(int row = 0;row<layer.getNeuronCount();row++)
			{
				for(int col = 0;col<synapse.getToNeuronCount();col++)
				{
					newMatrix.set(row,col,synapse.getMatrix().get(row, col));
				}
			}
			synapse.setMatrix(newMatrix);
		}
			
		// adjust the inbound weight matrixes
		Collection<Synapse> inboundSynapses = this.network.getStructure().getPreviousSynapses(layer);
		
		for(Synapse synapse: inboundSynapses)
		{
			Matrix newMatrix = new Matrix(synapse.getFromNeuronCount(), neuronCount);
			// copy existing matrix to new matrix
			for(int row = 0;row<synapse.getFromNeuronCount();row++)
			{
				for(int col = 0;col<synapse.getToNeuronCount();col++)
				{
					newMatrix.set(row,col,synapse.getMatrix().get(row, col));
				}
			}
			synapse.setMatrix(newMatrix);
		}
		
		// adjust the thresholds
		double[] newThresholds = new double[neuronCount];
		
		for(int i=0;i<layer.getNeuronCount();i++)
		{
			newThresholds[i] = layer.getThreshold(i);
		}
		
		layer.setThreshold(newThreshold);
		
		// finally, up the neuron count
		layer.setNeuronCount(neuronCount);
	}
	
	private void decreaseNeuronCount(Layer layer,int neuronCount)
	{
		// create an array to hold the least significant neurons, which will be removed
		int lostNeuronCount = layer.getNeuronCount() - neuronCount;
		double[] lostNeuronSignificance = new double[lostNeuronCount];
		int[] lostNeuron = new int[lostNeuronCount];
		
		// init the potential lost neurons to the first ones, we will find better choices if we can
		for(int i=0;i<lostNeuronCount;i++)
		{
			lostNeuron[i] = i;
			lostNeuronSignificance[i] = determineNeuronSignificance(layer,i);
		}
		
		// now loop over the remaining neurons and see if any are better ones to remove
		for(int i=lostNeuronCount;i<layer.getNeuronCount();i++)
		{
			double significance = determineNeuronSignificance(layer,i);
			
			// is this neuron less significant than one already chosen?
			for(int j = 0; j < lostNeuronCount; j++)
			{
				if( lostNeuronSignificance[j]>significance )
				{
					lostNeuron[j] = i;
					lostNeuronSignificance[j] = significance;
					break;
				}
			}
		}
		
		// finally, actually prune the neurons that the previous steps determined to remove
		for(int i=0;i<lostNeuronCount;i++)
		{
			prune(layer,lostNeuron[i]-i);
		}
		
	}
	
	/**
	 * Prune one of the neurons from this layer. Remove all entries in this
	 * weight matrix and other layers.
	 * 
	 * @param neuron
	 *            The neuron to prune. Zero specifies the first neuron.
	 */
	public void prune(final Layer targetLayer, final int neuron) {
		// delete a row on this matrix
		for (Synapse synapse : targetLayer.getNext()) {
			synapse
					.setMatrix(MatrixMath
							.deleteRow(synapse.getMatrix(), neuron));
		}

		// delete a column on the previous
		final Collection<Layer> previous = this.network.getStructure()
				.getPreviousLayers(targetLayer);

		for (Layer prevLayer : previous) {
			if (previous != null) {
				for (Synapse synapse : prevLayer.getNext()) {
					synapse.setMatrix(MatrixMath.deleteCol(synapse.getMatrix(),
							neuron));
				}
			}
		}
		
		// remove the threshold
		double[] newThreshold = new double[targetLayer.getNeuronCount() - 1];
		
		int targetIndex = 0;
		for(int i=0;i<targetLayer.getNeuronCount();i++)
		{
			if( targetIndex!=neuron )
			{
				newThreshold[targetIndex++] = targetLayer.getThreshold(i);
			}
		}
		
		targetLayer.setThreshold(newThreshold);
		
		// update the neuron count

		targetLayer.setNeuronCount(targetLayer.getNeuronCount() - 1);

	}

	public BasicNetwork getNetwork()
	{
		return this.network;
	}
	
	public double determineNeuronSignificance(Layer layer, int neuron)
	{
		// calculate the threshold significance
		double result = layer.getThreshold(neuron);
		
		// calculate the outbound significance
		for(Synapse synapse: layer.getNext())
		{
			for(int i=0;i<synapse.getToNeuronCount();i++)
			{
				result+=synapse.getMatrix().get(neuron, i);
			}
		}
		
		// calculate the threshold significance
		Collection<Synapse> inboundSynapses = this.network.getStructure().getPreviousSynapses(layer);
		
		for(Synapse synapse: inboundSynapses)
		{
			for(int i=0;i<synapse.getFromNeuronCount();i++)
			{
				result+=synapse.getMatrix().get(i, neuron);
			}
		}
		
		return Math.abs(result);
	}
	
	
	
}

