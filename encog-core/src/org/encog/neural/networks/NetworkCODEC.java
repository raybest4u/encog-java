package org.encog.neural.networks;

import java.util.Collection;

import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.synapse.Synapse;

public class NetworkCODEC {

	/**
	 * Private constructor.
	 */
	private NetworkCODEC() {

	}

	/**
	 * Use an array to populate the memory of the neural network.
	 * 
	 * @param array
	 *            An array of doubles.
	 * @param network
	 *            The network to encode.
	 */
	public static void arrayToNetwork(final Double[] array,
			final BasicNetwork network) {

		// copy all weight and threshold data
		int currentIndex = 0;
		Collection<Synapse> synapses = network.getSynapses();
		for (final Synapse synapse : synapses) {
			if (synapse.getMatrix() != null) {

					currentIndex = synapse.getMatrix().fromPackedArray(array,
							currentIndex);
				
			}
		}

	}


	/**
	 * Convert to an array. This is used with some training algorithms that
	 * require that the "memory" of the neuron(the weight and threshold values)
	 * be expressed as a linear array.
	 * 
	 * @param network
	 *            The network to encode.
	 * @return The memory of the neuron.
	 */
	public static Double[] networkToArray(final BasicNetwork network) {
		int size = 0;

		// first determine size
		for (final Synapse synapse : network.getSynapses()) {
			size += synapse.getMatrixSize();
		}

		// allocate an array to hold
		final Double[] result = new Double[size];

		// copy all weight and threshold data
		int currentIndex = 0;
		Collection<Synapse> synapses = network.getSynapses();
		for (final Synapse synapse : synapses ) {
			if (synapse.getMatrix() != null) {
				Double[] temp = synapse.getMatrix().toPackedArray();
				for (int i = 0; i < temp.length; i++) {
					result[currentIndex++] = temp[i];
				}
			}
		}

		return result;
	}


}
