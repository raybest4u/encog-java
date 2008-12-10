/*
 * Encog Workbench v1.x
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008, Heaton Research Inc., and individual contributors.
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
package org.encog.workbench.dialogs.training.backpropagation;

import java.awt.Frame;

import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.Train;
import org.encog.neural.networks.training.backpropagation.Backpropagation;
import org.encog.workbench.dialogs.training.BasicTrainingProgress;

/**
 * Dialog box to display the progress of backpropagation training.
 */
public class ProgressBackpropagation extends BasicTrainingProgress {

	/**
	 * The serial ID.
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * The learning rate.
	 */
	private final double learningRate;
	
	/**
	 * The momentum.
	 */
	private final double momentum;

	/**
	 * Construct the dialog box.
	 * @param owner The owner.
	 * @param network The network to train.
	 * @param trainingData The learning data to use for training.
	 * @param learningRate The learning rate.
	 * @param momentum The momentum.
	 * @param maxError The maximum error.
	 */
	public ProgressBackpropagation(final Frame owner,
			final BasicNetwork network, final NeuralDataSet trainingData,
			final double learningRate, final double momentum,
			final double maxError) {
		super(owner);
		setTitle("Backpropagation Training");
		setNetwork(network);
		setTrainingData(trainingData);
		this.learningRate = learningRate;
		this.momentum = momentum;
		setMaxError(maxError);

	}

	/**
	 * Perform one training iteration.
	 */
	@Override
	public void iteration() {

		getTrain().iteration();

	}

	/**
	 * Not used.
	 */
	@Override
	public void shutdown() {
	}

	/**
	 * Construct the training objects.
	 */
	@Override
	public void startup() {
		final Train train = new Backpropagation(getNetwork(),
				getTrainingData(), this.learningRate, this.momentum);

		setTrain(train);
	}

}
