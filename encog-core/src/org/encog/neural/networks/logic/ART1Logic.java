package org.encog.neural.networks.logic;

import org.encog.neural.data.NeuralData;
import org.encog.neural.data.bipolar.BiPolarNeuralData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.NeuralOutputHolder;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.synapse.Synapse;
import org.encog.neural.pattern.ART1Pattern;

public class ART1Logic extends ARTLogic {

	private Layer layerF1;
	private Layer layerF2;
	private Synapse synapseF1toF2;
	private Synapse synapseF2toF1;

	/**
	 * last winner in F2 layer.
	 */
	private int winner;

	/**
	 * A parameter for F1 layer.
	 */
	private double a1 = 1;

	/**
	 * B parameter for F1 layer.
	 */
	private double b1 = 1.5;

	/**
	 * C parameter for F1 layer.
	 */
	private double c1 = 5;

	/**
	 * D parameter for F1 layer.
	 */
	private double d1 = 0.9;

	/**
	 * L parameter for net.
	 */
	private double l = 3;

	/**
	 * The vigilance parameter.
	 */
	private double vigilance = 0.9;

	private boolean[] inhibitF1;
	private boolean[] inhibitF2;

	private int noWinner;

	private BiPolarNeuralData outputF1;
	private BiPolarNeuralData outputF2;

	
	private void init()
	{
		this.layerF1 = this.getNetwork().getInputLayer();
		this.layerF2 = this.getNetwork().getOutputLayer();
		this.inhibitF1 = new boolean[getNetwork().getInputLayer().getNeuronCount()];
		this.inhibitF2 = new boolean[getNetwork().getOutputLayer().getNeuronCount()];
		this.synapseF1toF2 = this.getNetwork().getStructure().findSynapse(this.layerF1, this.layerF2,true);
		this.synapseF2toF1 = this.getNetwork().getStructure().findSynapse(this.layerF2, this.layerF1,true);
		this.outputF1 = new BiPolarNeuralData(getNetwork().getInputLayer().getNeuronCount());
		this.outputF2 = new BiPolarNeuralData(getNetwork().getOutputLayer().getNeuronCount());

		this.a1 = 1;
		this.b1 = 1.5;
		this.c1 = 5;
		this.d1 = 0.9;
		this.l = 3;
		this.vigilance = 0.9;

		this.noWinner = getNetwork().getOutputLayer().getNeuronCount();
		reset();

	}
	
	public void reset()
	{
		for (int i = 0; i < this.layerF1.getNeuronCount(); i++) {
			for (int j = 0; j < this.layerF2.getNeuronCount(); j++) {
				this.synapseF1toF2.getMatrix().set(i, j,
						(this.b1 - 1) / this.d1 + 0.2);
				this.synapseF2toF1.getMatrix().set(j, i,
						this.l / (this.l - 1 + this.layerF1.getNeuronCount()) - 0.1);
			}
		}	
	}

	public void adjustWeights() {
		double magnitudeInput;

		for (int i = 0; i < this.layerF1.getNeuronCount(); i++) {
			if (this.outputF1.getBoolean(i)) {
				magnitudeInput = magnitude(this.outputF1);
				this.synapseF1toF2.getMatrix().set(i, this.winner, 1);
				this.synapseF2toF1.getMatrix().set(this.winner, i,
						this.l / (this.l - 1 + magnitudeInput));
			} else {
				this.synapseF1toF2.getMatrix().set(i, this.winner, 0);
				this.synapseF2toF1.getMatrix().set(this.winner, i, 0);
			}
		}
	}

	public void compute(final BiPolarNeuralData input,
			final BiPolarNeuralData output) {
		int i;
		boolean resonance, exhausted;
		double magnitudeInput1, magnitudeInput2;

		for (i = 0; i < this.layerF2.getNeuronCount(); i++) {
			this.inhibitF2[i] = false;
		}
		resonance = false;
		exhausted = false;
		do {
			setInput(input);
			computeF2();
			getOutput(output);
			if (this.winner != this.noWinner) {
				computeF1(input);
				magnitudeInput1 = magnitude(input);
				magnitudeInput2 = magnitude(this.outputF1);
				if ((magnitudeInput2 / magnitudeInput1) < this.vigilance) {
					this.inhibitF2[this.winner] = true;
				} else {
					resonance = true;
				}
			} else {
				exhausted = true;
			}
		} while (!(resonance || exhausted));
		if (resonance) {
			adjustWeights();
		}
	}

	private void computeF1(final BiPolarNeuralData input) {
		double sum, activation;

		for (int i = 0; i < this.layerF1.getNeuronCount(); i++) {
			sum = this.synapseF1toF2.getMatrix().get(i, this.winner)
					* (this.outputF2.getBoolean(this.winner) ? 1 : 0);
			activation = ((input.getBoolean(i) ? 1 : 0) + this.d1 * sum - this.b1)
					/ (1 + this.a1
							* ((input.getBoolean(i) ? 1 : 0) + this.d1 * sum) + this.c1);
			this.outputF1.setData(i, activation > 0);
		}
	}

	private void computeF2() {
		int i, j;
		double Sum, maxOut;

		maxOut = Double.NEGATIVE_INFINITY;
		this.winner = this.noWinner;
		for (i = 0; i < this.layerF2.getNeuronCount(); i++) {
			if (!this.inhibitF2[i]) {
				Sum = 0;
				for (j = 0; j < this.layerF1.getNeuronCount(); j++) {
					Sum += this.synapseF2toF1.getMatrix().get(i, j)
							* (this.outputF1.getBoolean(j) ? 1 : 0);
				}
				if (Sum > maxOut) {
					maxOut = Sum;
					this.winner = i;
				}
			}
			this.outputF2.setData(i, false);
		}
		if (this.winner != this.noWinner) {
			this.outputF2.setData(this.winner, true);
		}
	}

	public double getA1() {
		return this.a1;
	}

	public double getB1() {
		return this.b1;
	}

	public double getC1() {
		return this.c1;
	}

	public double getD1() {
		return this.d1;
	}

	public double getL() {
		return this.l;
	}

	private void getOutput(final BiPolarNeuralData output) {
		for (int i = 0; i < this.layerF2.getNeuronCount(); i++) {
			output.setData(i, this.outputF2.getBoolean(i));
		}
	}

	public double getVigilance() {
		return this.vigilance;
	}

	public int getWinner() {
		return this.winner;
	}

	public boolean hasWinner() {
		return this.winner != this.noWinner;
	}

	public double magnitude(final BiPolarNeuralData input) {
		double result;

		result = 0;
		for (int i = 0; i < this.layerF1.getNeuronCount(); i++) {
			result += input.getBoolean(i) ? 1 : 0;
		}
		return result;
	}

	public void setA1(final double a1) {
		this.a1 = a1;
	}

	public void setB1(final double b1) {
		this.b1 = b1;
	}

	public void setC1(final double c1) {
		this.c1 = c1;
	}

	public void setD1(final double d1) {
		this.d1 = d1;
	}

	private void setInput(final BiPolarNeuralData input) {
		double activation;

		for (int i = 0; i < this.layerF1.getNeuronCount(); i++) {
			activation = (input.getBoolean(i) ? 1 : 0)
					/ (1 + this.a1 * ((input.getBoolean(i) ? 1 : 0) + this.b1) + this.c1);
			this.outputF1.setData(i, (activation > 0));
		}
	}

	public void setL(final double l) {
		this.l = l;
	}

	public void setVigilance(final double vigilance) {
		this.vigilance = vigilance;
	}
	
	public void init(BasicNetwork network)
	{
		super.init(network);
		
		this.layerF1 = this.getNetwork().getInputLayer();
		this.layerF2 = this.getNetwork().getOutputLayer();
		this.inhibitF1 = new boolean[network.getInputLayer().getNeuronCount()];
		this.inhibitF2 = new boolean[network.getOutputLayer().getNeuronCount()];
		this.synapseF1toF2 = this.getNetwork().getStructure().findSynapse(this.layerF1, this.layerF2,true);
		this.synapseF2toF1 = this.getNetwork().getStructure().findSynapse(this.layerF2, this.layerF1,true);
		this.outputF1 = new BiPolarNeuralData(network.getInputLayer().getNeuronCount());
		this.outputF2 = new BiPolarNeuralData(network.getOutputLayer().getNeuronCount());

		this.a1 = this.getNetwork().getPropertyDouble(ARTLogic.PROPERTY_A1);
		this.b1 = this.getNetwork().getPropertyDouble(ARTLogic.PROPERTY_B1);
		this.c1 = this.getNetwork().getPropertyDouble(ARTLogic.PROPERTY_C1);
		this.d1 = this.getNetwork().getPropertyDouble(ARTLogic.PROPERTY_D1);
		this.l = this.getNetwork().getPropertyDouble(ARTLogic.PROPERTY_L);
		this.vigilance = this.getNetwork().getPropertyDouble(ARTLogic.PROPERTY_VIGILANCE);

		this.noWinner = network.getOutputLayer().getNeuronCount();
		reset();

	}
	
	
	@Override
	public NeuralData compute(NeuralData input, NeuralOutputHolder useHolder) {
		// TODO Auto-generated method stub
		return null;
	}

}
