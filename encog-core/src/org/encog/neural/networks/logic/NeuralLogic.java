package org.encog.neural.networks.logic;

import java.io.Serializable;

import org.encog.neural.data.NeuralData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.NeuralOutputHolder;

public interface NeuralLogic extends Serializable {
	
	public NeuralData compute(NeuralData input,
			NeuralOutputHolder useHolder);
	public void init(BasicNetwork network);
}
