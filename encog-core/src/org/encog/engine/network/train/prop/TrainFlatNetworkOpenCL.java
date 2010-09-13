package org.encog.engine.network.train.prop;

import java.util.HashMap;
import java.util.Map;

import org.encog.engine.EncogEngine;
import org.encog.engine.EncogEngineError;
import org.encog.engine.data.EngineDataSet;
import org.encog.engine.data.EngineIndexableSet;
import org.encog.engine.network.flat.FlatNetwork;
import org.encog.engine.network.train.TrainFlatNetwork;
import org.encog.engine.opencl.EncogCLDevice;
import org.encog.engine.opencl.EncogCLPlatform;
import org.encog.engine.opencl.kernels.KernelNetworkTrain;
import org.encog.engine.opencl.kernels.TrainingWorkload;
import org.encog.engine.util.ErrorCalculation;
import org.encog.engine.util.ErrorCalculationMode;

public class TrainFlatNetworkOpenCL implements TrainFlatNetwork {

	private double error;
	private EncogCLDevice targetDevice;

	/**
	 * The network to train.
	 */
	private final FlatNetwork network;

	/**
	 * The gradients.
	 */
	private final double[] gradients;

	/**
	 * The weights and thresholds.
	 */
	private final double[] weights;

	/**
	 * The training data.
	 */
	private final EngineIndexableSet training;

	/**
	 * THe workload to use.
	 */
	private final TrainingWorkload workload;

	
	/**
	 * Train a flat network multithreaded.
	 * 
	 * @param network
	 *            The network to train.
	 * @param training
	 *            The training data to use.
	 */
	public TrainFlatNetworkOpenCL(final FlatNetwork network,
			final EngineDataSet training, final EncogCLDevice targetDevice) {

		if (!(training instanceof EngineIndexableSet)) {
			throw new EncogEngineError(
					"Training data must be Indexable for this training type.");
		}
		
		if (EncogEngine.getInstance().getCL() != null) {
			throw new EncogEngineError(
			"You must enable OpenCL before using this training type.");

		}
		
		this.targetDevice = targetDevice;
		this.network = network;
		this.training = (EngineIndexableSet)training;

		this.gradients = new double[network.getWeights().length];

		this.weights = network.getWeights();


		this.workload = new TrainingWorkload(
				this.targetDevice, 
				network, 
				this.training, 
				(int)this.training.getRecordCount(),
				0);
		

			final Map<String, String> options = new HashMap<String, String>();
			options.put("NEURON_COUNT", "" + this.network.getNeuronCount());
			options.put("WEIGHT_COUNT", "" + this.network.getWeights().length);

			for (final EncogCLPlatform platform : EncogEngine.getInstance()
					.getCL().getPlatforms()) {
				platform.getNetworkTrain().compile(options);
				platform.getNetworkTrain().init(this.network);
			}

	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public double getError() {
		return error;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public FlatNetwork getNetwork() {
		return this.network;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getNumThreads() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * {@inheritDoc}
	 */
	public EncogCLDevice getTargetDevice() {
		return this.targetDevice;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public EngineDataSet getTraining() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void iteration() {

		final KernelNetworkTrain k = this.targetDevice.getPlatform()
				.getNetworkTrain();

		k.calculate(this.workload);

		for (int j = 0; j < this.gradients.length; j++) {
			this.gradients[j] = 0;
		}

		double e = 0;
		int index = 0;
		int errorIndex = 0;

		for (int i = 0; i < this.workload.getMaxUnits(); i++) {
			e += this.workload.getErrors()[errorIndex++];

			for (int j = 0; j < this.gradients.length; j++) {
				this.gradients[j] += this.workload.getGradients()[index++];
			}
		}

		final int count = (int)this.training.getRecordCount();
		
		this.error = e
				/ (count * this.training.getIdealSize());
		
		if( ErrorCalculation.getMode()==ErrorCalculationMode.RMS) {
			this.error = Math.sqrt(error);
		}
		//this.owner.report(this.gradients, error, null);


		
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void setNumThreads(int numThreads) {
		// TODO Auto-generated method stub
		
	}


	
}
