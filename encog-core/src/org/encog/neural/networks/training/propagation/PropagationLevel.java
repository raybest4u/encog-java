package org.encog.neural.networks.training.propagation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.encog.neural.NeuralNetworkError;
import org.encog.neural.activation.ActivationFunction;
import org.encog.neural.data.NeuralData;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.synapse.Synapse;


public class PropagationLevel {
	
	private final int neuronCount;
	private final List<Layer> layers = new ArrayList<Layer>();
	private final List<PropagationSynapse> outgoing = new ArrayList<PropagationSynapse>();
	private final double[] deltas;
	private final Propagation propagation;
	
	public PropagationLevel(Propagation propagation, Layer layer)
	{
		this.neuronCount = layer.getNeuronCount();
		this.deltas = new double[this.neuronCount];
		this.layers.add(layer);		
		this.propagation = propagation;
	}
	
	public PropagationLevel(Propagation propagation, List<Synapse> outgoing)
	{
		int count = 0;
		
		this.propagation = propagation;
		this.outgoing.clear();
		
		Set<Layer> layerSet = new HashSet<Layer>();
		
		for(Synapse synapse: outgoing)
		{
			count+=synapse.getFromNeuronCount();
			layerSet.add(synapse.getFromLayer());
			PropagationSynapse propSynapse = new PropagationSynapse(synapse);
			this.outgoing.add(propSynapse);
		}
		
		this.layers.addAll(layerSet);
		
		this.neuronCount = count;
		
		this.deltas = new double[this.neuronCount];
	}

	public int getNeuronCount() {
		return neuronCount;
	}

	public List<PropagationSynapse> getOutgoing() {
		return outgoing;
	}

	public void setDelta(int index, double d) {
		this.deltas[index] = d;		
	}
	
	public double getDelta(int index)
	{
		return this.deltas[index];
	}
	
	public double[] getDeltas()
	{
		return this.deltas;
	}

	public List<Layer> getLayers() {
		return layers;
	}
	
	public List<Synapse> determinePreviousSynapses()
	{
		List<Synapse> result = new ArrayList<Synapse>();
		
		for(Layer layer: this.layers)
		{
			Collection<Synapse> synapses = this.propagation.getNetwork().getStructure().getPreviousSynapses(layer);
			result.addAll(synapses);
		}
			
		return result;
		
	}
	

	public double getActual(final int index)
	{
		int currentIndex = index;
		
		// is this the output layer, if so then we need to return the output from
		// the entire network.
		if( this.outgoing.size()==0 )
		{
			NeuralData actual = this.propagation.getOutputHolder().getOutput();
			return actual.getData(index);
		}
		
		// not the output layer, so we need output from one of the previous layers.
		for(PropagationSynapse synapse: this.outgoing )
		{
			int count = synapse.getSynapse().getFromNeuronCount();
			
			if( currentIndex<count )
			{
				NeuralData actual = this.propagation.getOutputHolder().getResult().get(synapse.getSynapse());
				return actual.getData(currentIndex);
			}
			
			currentIndex-=count;
		}
		
		throw new NeuralNetworkError("Could not find actual value while propagation training.");
	}


	public void learn() {
		for(PropagationSynapse synapse: this.outgoing)
		{
			synapse.learn(this.propagation.getLearningRate(), this.propagation.getMomentum());
		}		
	}

	public void applyDerivative() {
				
		// performance shortcut
		if( this.layers.size()==0 )
		{
			Layer layer = layers.get(0);
			layer.getActivationFunction().derivativeFunction(this.deltas);
		}
		else
		{
			int currentIndex = 0;
			double[] temp = new double[1];
			
			// multiple layers, so maybe multiple activation functions
			for(Layer layer: this.layers)
			{
				// obtain this layer's activation function
				ActivationFunction function = layer.getActivationFunction();
				
				// apply the activation function to all delta's covered by it
				for(int i=0;i<layer.getNeuronCount();i++)
				{
					temp[0] = this.deltas[currentIndex];
					function.derivativeFunction(temp);
					this.deltas[currentIndex++] = temp[0];
				}
			}			
		}
	}	
}
