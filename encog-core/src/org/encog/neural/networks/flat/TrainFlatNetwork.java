package org.encog.neural.networks.flat;

import org.encog.mathutil.error.ErrorCalculation;
import org.encog.neural.data.NeuralDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class TrainFlatNetwork {
	
	private NeuralDataSet training;
	private FlatNetwork network;
	private Object[] layerDelta;
	private double[] gradients;
	private double[] lastGradient;
	private ErrorCalculation errorCalculation = new ErrorCalculation();
	private double[] updateValues;
	
	public TrainFlatNetwork(FlatNetwork network, NeuralDataSet training)
	{
		this.training = training;
		this.network = network;
		
		this.layerDelta = new Object[network.getLayerCounts().length];
		this.gradients = new double[network.getWeights().length];
		this.updateValues = new double[network.getWeights().length];
		this.lastGradient = new double[network.getWeights().length];
		
		for(int i=0;i<this.updateValues.length;i++)
		{
			this.updateValues[i] = ResilientPropagation.DEFAULT_INITIAL_UPDATE;
		}
		
		for(int i=0;i<network.getLayerCounts().length;i++)
		{
			this.layerDelta[i] = new double[network.getLayerCounts()[i]];
		}
	}
	
	public double derivativeFunction(final double d) {
		return d * (1.0 - d);
	}
	
	public void iteration()
	{
		errorCalculation.reset();
		
		for(NeuralDataPair pair: this.training)
		{
			double[] input = pair.getInput().getData();
			double[] ideal = pair.getIdeal().getData();
			double[] actual = new double[network.getOutputCount()];
			
			this.network.calculate(input,actual);
			
			errorCalculation.updateError(actual, ideal);
			
			double[] error = (double[])this.layerDelta[0];
			
			for(int i=0;i<actual.length;i++)
			{
				error[i] = derivativeFunction(actual[i])*(ideal[i]-actual[i]);
			}
			
			for(int i=0;i<this.network.getLayerCounts().length-1;i++)
			{
				processLevel(i);
			}
		}
		
		learn();
	}
	
	private void processLevel(int currentLevel)
	{
		double[] fromDeltas = (double[])layerDelta[currentLevel+1];
		double[] toDeltas = (double[])layerDelta[currentLevel];
		
		// clear the to-deltas
		for(int i=0;i<fromDeltas.length;i++)
		{
			fromDeltas[i] = 0;
		}
		
		int index = this.network.getWeightIndex()[currentLevel]+toDeltas.length;
		int layerIndex = this.network.getLayerIndex()[currentLevel+1];
		int layerSize = this.network.getLayerCounts()[currentLevel+1];

		for (int x = 0; x < toDeltas.length; x++) {
			for (int y = 0; y < fromDeltas.length; y++) {
				final double value = network.getLayerOutput()[layerIndex+y] * toDeltas[x];
				this.gradients[index] += value;
				fromDeltas[y] +=  this.network.getWeights()[index] * toDeltas[x];
				index++;
			}
		}

		for (int i = 0; i < layerSize
		; i++) {
			fromDeltas[i]*= this.derivativeFunction(this.network.getLayerOutput()[layerIndex+i]);
		}
	}
	
	private void learn()
	{
		for (int i = 0; i < this.gradients.length; i++) {
			this.network.getWeights()[i]+=updateWeight(this.gradients, i);
			this.gradients[i] = 0;
		}
	}
	
	/**
	 * Determine the amount to change a weight by.
	 * @param gradients The gradients.
	 * @param index The weight to adjust.
	 * @return The amount to change this weight by.
	 */
	private double updateWeight(final double[] gradients, final int index) {
		// multiply the current and previous gradient, and take the
		// sign. We want to see if the gradient has changed its sign.
		final int change = sign(this.gradients[index]
				* this.lastGradient[index]);
		double weightChange = 0;

		// if the gradient has retained its sign, then we increase the
		// delta so that it will converge faster
		if (change > 0) {
			double delta = this.updateValues[index]
					* ResilientPropagation.POSITIVE_ETA;
			delta = Math.min(delta, ResilientPropagation.DEFAULT_MAX_STEP);
			weightChange = sign(this.gradients[index]) * delta;
			this.updateValues[index] = delta;
			this.lastGradient[index] = this.gradients[index];
		} else if (change < 0) {
			// if change<0, then the sign has changed, and the last
			// delta was too big
			double delta = this.updateValues[index]
					* ResilientPropagation.NEGATIVE_ETA;
			delta = Math.max(delta, ResilientPropagation.DELTA_MIN);
			this.updateValues[index] = delta;
			// set the previous gradent to zero so that there will be no
			// adjustment the next iteration
			this.lastGradient[index] = 0;
		} else if (change == 0) {
			// if change==0 then there is no change to the delta
			final double delta = this.lastGradient[index];
			weightChange = sign(this.gradients[index]) * delta;
			this.lastGradient[index] = this.gradients[index];
		}

		// apply the weight change, if any
		return weightChange;
	}
	
	/**
	 * Determine the sign of the value.
	 * 
	 * @param value
	 *            The value to check.
	 * @return -1 if less than zero, 1 if greater, or 0 if zero.
	 */
	private int sign(final double value) {
		if (Math.abs(value) < ResilientPropagation.DEFAULT_ZERO_TOLERANCE) {
			return 0;
		} else if (value > 0) {
			return 1;
		} else {
			return -1;
		}
	}
	
	public double getError()
	{
		return errorCalculation.calculateRMS();
	}
	
}
