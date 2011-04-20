/*
 * Encog(tm) Core v3.0 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package org.encog.neural.networks.training.propagation;

import org.encog.EncogError;
import org.encog.ml.MLMethod;
import org.encog.ml.TrainingImplementationType;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.flat.train.TrainFlatNetwork;
import org.encog.neural.networks.ContainsFlat;
import org.encog.neural.networks.training.TrainingError;
import org.encog.util.EncogValidate;
import org.encog.util.logging.EncogLogging;

/**
 * Implements basic functionality that is needed by each of the propagation
 * methods. The specifics of each of the propagation methods is implemented
 * inside of the PropagationMethod interface implementors.
 * 
 * @author jheaton
 * 
 */
public abstract class Propagation extends BasicTraining {

	/**
	 * The network.
	 */
	private final ContainsFlat network;

	/**
	 * The current flat network we are using for training, or null for none.
	 */
	private FlatNetwork currentFlatNetwork;

	/**
	 * The current flat trainer we are using, or null for none.
	 */
	private TrainFlatNetwork flatTraining;


	/**
	 * Construct a propagation object.
	 * 
	 * @param network
	 *            The network.
	 * @param training
	 *            The training set.
	 */
	public Propagation(final ContainsFlat network, 
			final MLDataSet training) {
		super(TrainingImplementationType.Iterative);
		this.network = network;
		setTraining(training);
	}

	/**
	 * @return True if this training can be continued.
	 */
	public boolean canContinue() {
		return false;
	}

	/**
	 * Should be called after training has completed and the iteration method
	 * will not be called any further.
	 */
	@Override
	public void finishTraining() {
		super.finishTraining();
		this.flatTraining.finishTraining();
	}

	/**
	 * @return the currentFlatNetwork
	 */
	public FlatNetwork getCurrentFlatNetwork() {
		return this.currentFlatNetwork;
	}

	/**
	 * @return the flatTraining
	 */
	public TrainFlatNetwork getFlatTraining() {
		return this.flatTraining;
	}

	/**
	 * {@inheritDoc}
	 */
	public MLMethod getMethod() {
		return this.network;
	}

	/**
	 * @return The number of threads.
	 */
	public int getNumThreads() {
		return this.flatTraining.getNumThreads();
	}

	/**
	 * Determine if this specified training continuation object is valid for
	 * this training method.
	 * 
	 * @param state
	 *            The training continuation object to check.
	 * @return True if the continuation object is valid.
	 */
	public boolean isValidResume(final TrainingContinuation state) {
		return false;
	}

	/**
	 * Perform one training iteration.
	 */
	public void iteration() {
		try {		
			preIteration();

			this.flatTraining.iteration();
			setError(this.flatTraining.getError());

			postIteration();

			EncogLogging.log(EncogLogging.LEVEL_INFO,"Training iteration done, error: "
						+ getError());			
		} catch (final ArrayIndexOutOfBoundsException ex) {
			EncogValidate.validateNetworkForTraining(this.network,
					getTraining());
			throw new EncogError(ex);
		}
	}

	/**
	 * Perform the specified number of training iterations. This can be more
	 * efficient than single training iterations. This is particularly true if
	 * you are training with a GPU.
	 * 
	 * @param count
	 *            The number of training iterations.
	 */
	@Override
	public void iteration(final int count) {
		try {
			preIteration();

			this.flatTraining.iteration(count);
			setIteration(this.flatTraining.getIteration());
			setError(this.flatTraining.getError());

			postIteration();

			EncogLogging.log(EncogLogging.LEVEL_INFO,"Training iterations done, error: "
						+ getError());
			
		} catch (final ArrayIndexOutOfBoundsException ex) {
			EncogValidate.validateNetworkForTraining(this.network,
					getTraining());
			throw new EncogError(ex);
		}
	}

	/**
	 * Pause the training to continue later.
	 * 
	 * @return A training continuation object.
	 */
	public TrainingContinuation pause() {
		throw new TrainingError("This training type does not support pause.");
	}

	/**
	 * Resume training.
	 * 
	 * @param state
	 *            The training continuation object to use to continue.
	 */
	public void resume(final TrainingContinuation state) {
		throw new TrainingError("This training type does not support resume.");
	}

	/**
	 * @param flatTraining
	 *            the flatTraining to set
	 */
	public void setFlatTraining(final TrainFlatNetwork flatTraining) {
		this.flatTraining = flatTraining;
	}

	/**
	 * Set the number of threads. Specify zero to tell Encog to automatically
	 * determine the best number of threads for the processor. If OpenCL is used
	 * as the target device, then this value is not used.
	 * 
	 * @param numThreads
	 *            The number of threads.
	 */
	public void setNumThreads(final int numThreads) {
		this.flatTraining.setNumThreads(numThreads);
	}

}
