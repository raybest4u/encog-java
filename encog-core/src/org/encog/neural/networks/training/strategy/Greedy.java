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
package org.encog.neural.networks.training.strategy;

import org.encog.ml.MLEncodable;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.structure.NetworkCODEC;
import org.encog.neural.networks.training.Strategy;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.TrainingError;
import org.encog.util.logging.EncogLogging;

/**
 * A simple greedy strategy. If the last iteration did not improve training,
 * then discard it. Care must be taken with this strategy, as sometimes a
 * training algorithm may need to temporarily decrease the error level before
 * improving it.
 * 
 * @author jheaton
 * 
 */
public class Greedy implements Strategy {

	/**
	 * The training algorithm that is using this strategy.
	 */
	private Train train;
	
	/**
	 * The error rate from the previous iteration.
	 */
	private double lastError;
	
	/**
	 * The last state of the network, so that we can restore to this
	 * state if needed.
	 */
	private double[] lastNetwork;
	
	/**
	 * Has one iteration passed, and we are now ready to start 
	 * evaluation.
	 */
	private boolean ready;

	private MLEncodable method;
	
	/**
	 * Initialize this strategy.
	 * @param train The training algorithm.
	 */
	public void init(final Train train) {
		this.train = train;
		this.ready = false;
		
		if( !(train.getNetwork() instanceof MLEncodable) ) {
			throw new TrainingError("To make use of the Greedy strategy the machine learning method must support MLEncodable.");
		}
		
		this.method = ((MLEncodable)train.getNetwork());
		this.lastNetwork = new double[this.method.encodedArrayLength()];
	}

	/**
	 * Called just after a training iteration.
	 */
	public void postIteration() {
		if (this.ready) {
			if (this.train.getError() > this.lastError) {
				EncogLogging.log(EncogLogging.LEVEL_DEBUG,"Greedy strategy dropped last iteration.");				
				this.train.setError(this.lastError);
				this.method.decodeFromArray(this.lastNetwork);
			}
		} else {
			this.ready = true;
		}
	}

	/**
	 * Called just before a training iteration.
	 */
	public void preIteration() {

		if (this.method != null) {
			this.lastError = this.train.getError();
			this.method.encodeToArray(this.lastNetwork);
			this.train.setError(this.lastError);
		}
	}
}
