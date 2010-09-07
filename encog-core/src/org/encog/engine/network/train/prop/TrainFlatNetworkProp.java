/*
 * Encog(tm) Core v2.5 
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2010 by Heaton Research Inc.
 * 
 * Released under the LGPL.
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
 * 
 * Encog and Heaton Research are Trademarks of Heaton Research, Inc.
 * For information on Heaton Research trademarks, visit:
 * 
 * http://www.heatonresearch.com/copyright.html
 */

package org.encog.engine.network.train.prop;

import java.util.HashMap;
import java.util.Map;

import org.encog.engine.EncogEngine;
import org.encog.engine.EncogEngineError;
import org.encog.engine.concurrency.DetermineWorkload;
import org.encog.engine.concurrency.EngineConcurrency;
import org.encog.engine.concurrency.TaskGroup;
import org.encog.engine.data.EngineIndexableSet;
import org.encog.engine.data.EngineDataSet;
import org.encog.engine.network.flat.ActivationFunctions;
import org.encog.engine.network.flat.FlatNetwork;
import org.encog.engine.network.flat.ValidateForOpenCL;
import org.encog.engine.network.train.TrainFlatNetwork;
import org.encog.engine.network.train.gradient.FlatGradientWorker;
import org.encog.engine.network.train.gradient.GradientWorkerCL;
import org.encog.engine.network.train.gradient.GradientWorkerCPU;
import org.encog.engine.opencl.EncogCLDevice;
import org.encog.engine.opencl.EncogCLPlatform;
import org.encog.engine.util.EngineArray;
import org.encog.engine.util.IntRange;

/**
 * Train a flat network using multithreading, and GPU support.
 * 
 * The training data must be indexable, it will be broken into groups for each
 * thread to process.
 * 
 * At the end of each iteration the training from each thread is aggregated back
 * to the neural network.
 * 
 */
public abstract class TrainFlatNetworkProp implements TrainFlatNetwork {

	/**
	 * The number of threads to use.
	 */
	protected int numThreads;

	/**
	 * The gradients.
	 */
	protected double[] gradients;

	/**
	 * The last gradients, from the last training iteration.
	 */
	protected double[] lastGradient;

	/**
	 * The network to train.
	 */
	protected final FlatNetwork network;

	/**
	 * The training data.
	 */
	protected final EngineDataSet training;

	/**
	 * The network in indexable form.
	 */
	protected final EngineIndexableSet indexable;

	/**
	 * The workers.
	 */
	protected FlatGradientWorker[] workers;

	/**
	 * The total error. Used to take the average of.
	 */
	protected double totalError;

	/**
	 * The current error is the average error over all of the threads.
	 */
	protected double currentError;

	/**
	 * Reported exception from the threads.
	 */
	protected Throwable reportedException;

	/**
	 * The OpenCL device targeted.
	 */
	protected EncogCLDevice targetDevice;

	/**
	 * Train a flat network multithreaded.
	 * 
	 * @param network
	 *            The network to train.
	 * @param training
	 *            The training data to use.
	 */
	public TrainFlatNetworkProp(final FlatNetwork network,
			final EngineDataSet training) {

		if (!(training instanceof EngineIndexableSet)) {
			throw new EncogEngineError(
					"Training data must be Indexable for this training type.");
		}

		this.training = training;
		this.network = network;
		
		this.gradients = new double[this.network.getWeights().length];
		this.lastGradient = new double[this.network.getWeights().length];

		this.indexable = (EngineIndexableSet) training;
		this.numThreads = 0;
		this.reportedException = null;
	}

	/**
	 * Calculate the GPU vs CPU performance.
	 */
	private void calculatePerformance() {
		long totalCPU = 0;
		long countCPU = 0;
		long totalCL = 0;
		long countCL = 0;

		for (final FlatGradientWorker worker : this.workers) {
			if (worker instanceof GradientWorkerCPU) {
				countCPU++;
				totalCPU += worker.getElapsedTime();
			} else if (worker instanceof GradientWorkerCL) {
				countCL++;
				totalCL += worker.getElapsedTime();
			}
		}
	}

	/**
	 * Copy the contexts to keep them consistent with multithreaded training.
	 */
	private void copyContexts() {
		for (int i = 0; i < (this.workers.length - 1); i++) {
			final double[] src = this.workers[i].getNetwork().getLayerOutput();
			final double[] dst = this.workers[i + 1].getNetwork()
					.getLayerOutput();
			EngineArray.arrayCopy(src, dst);
		}
	}

	/**
	 * @return The error from the neural network.
	 */
	public double getError() {
		return this.currentError;
	}

	/**
	 * @return The trained neural network.
	 */
	public FlatNetwork getNetwork() {
		return this.network;
	}

	/**
	 * @return The target OpenCL device.
	 */
	public EncogCLDevice getTargetDevice() {
		return this.targetDevice;
	}

	/**
	 * @return The data we are training with.
	 */
	public EngineDataSet getTraining() {
		return this.training;
	}

	/**
	 * Init the process.
	 */
	private void init() {

		DetermineWorkload determine;
		final ValidateForOpenCL val = new ValidateForOpenCL();

		if ((this.targetDevice != null) && (val.isValid(this.network) == null)) {
			if (EncogEngine.getInstance().getCL().areCPUsPresent()) {
				this.numThreads = -1;
				EncogEngine.getInstance().getCL().enableAllCPUs();
			}

			determine = new DetermineWorkload(this.numThreads, 1,
					(int) this.indexable.getRecordCount());
		} else {
			determine = new DetermineWorkload(this.numThreads,
					(int) this.indexable.getRecordCount());
		}

		this.workers = new FlatGradientWorker[determine.getTotalWorkerCount()];

		determine.calculateWorkers();
		int index = 0;

		// if we are using CL, then we need to compile the kernels for this
		// network
		if (EncogEngine.getInstance().getCL() != null) {
			final Map<String, String> options = new HashMap<String, String>();
			options.put("NEURON_COUNT", "" + this.network.getNeuronCount());
			options.put("WEIGHT_COUNT", "" + this.network.getWeights().length);

			// is there only one activation function? If so, there are some
			// optimizations we can use.
			final int act = this.network.hasSameActivationFunction();

			if (act == ActivationFunctions.ACTIVATION_SIGMOID) {
				options.put("USE_SIGMOID", "1");
			} else if (act == ActivationFunctions.ACTIVATION_TANH) {
				options.put("USE_TANH", "1");
			}

			for (final EncogCLPlatform platform : EncogEngine.getInstance()
					.getCL().getPlatforms()) {
				platform.getNetworkTrain().compile(options);
				platform.getNetworkTrain().init(this.network);
			}
		}

		// handle CL, should only be one currently.
		// additional devices can be used, but only on different trainers.
		for (final IntRange r : determine.getCLRanges()) {
			this.workers[index++] = new GradientWorkerCL(this.targetDevice,
					this.network.clone(), this,
					this.indexable.openAdditional(), r.getLow(), r.getHigh());
		}

		// handle CPU
		for (final IntRange r : determine.getCPURanges()) {
			this.workers[index++] = new GradientWorkerCPU(this.network.clone(),
					this, this.indexable.openAdditional(), r.getLow(), r
							.getHigh());
		}
	}
	
	public void calculateGradients()
	{
		if (this.workers == null) {
			init();
		}

		this.workers[0].getNetwork().clearContext();
		this.totalError = 0;

		if (this.workers.length > 1) {

			final TaskGroup group = EngineConcurrency.getInstance()
					.createTaskGroup();

			for (final FlatGradientWorker worker : this.workers) {
				EngineConcurrency.getInstance().processTask(worker, group);
			}

			group.waitForComplete();
		} else {
			this.workers[0].run();
		}
		
		this.currentError = this.totalError / this.workers.length;

	}

	/**
	 * Perform one training iteration.
	 */
	public void iteration() {
		calculateGradients();
		learn();
		
		for (final FlatGradientWorker worker : this.workers) {
			EngineArray.arrayCopy(this.network.getWeights(), 0, worker
					.getWeights(), 0, this.network.getWeights().length);
		}

		copyContexts();

		if (this.reportedException != null) {
			throw (new EncogEngineError(this.reportedException));
		}

		calculatePerformance();
	}

	/**
	 * Apply and learn.
	 */
	protected void learn() {
		final double[] weights = this.network.getWeights();
		for (int i = 0; i < this.gradients.length; i++) {
			weights[i] += updateWeight(this.gradients, this.lastGradient, i);
			this.gradients[i] = 0;
		}
	}

	/**
	 * Called by the worker threads to report the progress at each step.
	 * 
	 * @param gradients
	 *            The gradients from that worker.
	 * @param error
	 *            The error for that worker.
	 * @param ex The exception.
	 */
	public void report(final double[] gradients, final double error,
			final Throwable ex) {
		synchronized (this) {
			if (ex == null) {

				for (int i = 0; i < gradients.length; i++) {
					this.gradients[i] += gradients[i];
				}
				this.totalError += error;
			} else {
				this.reportedException = ex;
			}
		}
	}

	/**
	 * Set the number of threads to use.
	 * 
	 * @param numThreads
	 *            The number of threads to use.
	 */
	public void setNumThreads(final int numThreads) {
		this.numThreads = numThreads;
	}

	/**
	 * Set the target device.
	 * @param targetDevice The target device.
	 */
	public void setTargetDevice(final EncogCLDevice targetDevice) {
		this.targetDevice = targetDevice;
	}

	/**
	 * Update a weight, the means by which weights are updated vary depending on
	 * the training.
	 * 
	 * @param gradients
	 *            The gradients.
	 * @param lastGradient
	 *            The last gradients.
	 * @param index
	 *            The index.
	 * @return The update value.
	 */
	public abstract double updateWeight(double[] gradients,
			double[] lastGradient, int index);

	/**
	 * @return The gradients from the last iteration;
	 */
	public double[] getLastGradient() {
		return lastGradient;
	}
	
	

}
