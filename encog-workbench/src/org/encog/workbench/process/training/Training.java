/*
 * Encog(tm) Workbench v2.4
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

package org.encog.workbench.process.training;

import java.awt.Frame;

import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.logic.SOMLogic;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.dialogs.training.ChooseTrainingMethodDialog;
import org.encog.workbench.dialogs.training.adaline.InputAdaline;
import org.encog.workbench.dialogs.training.adaline.ProgressAdaline;
import org.encog.workbench.dialogs.training.anneal.InputAnneal;
import org.encog.workbench.dialogs.training.anneal.ProgressAnneal;
import org.encog.workbench.dialogs.training.backpropagation.InputBackpropagation;
import org.encog.workbench.dialogs.training.backpropagation.ProgressBackpropagation;
import org.encog.workbench.dialogs.training.genetic.InputGenetic;
import org.encog.workbench.dialogs.training.genetic.ProgressGenetic;
import org.encog.workbench.dialogs.training.hopfield.InputHopfield;
import org.encog.workbench.dialogs.training.instar.InputInstar;
import org.encog.workbench.dialogs.training.instar.ProgressInstar;
import org.encog.workbench.dialogs.training.lma.InputLMA;
import org.encog.workbench.dialogs.training.lma.ProgressLMA;
import org.encog.workbench.dialogs.training.manhattan.InputManhattan;
import org.encog.workbench.dialogs.training.manhattan.ProgressManhattan;
import org.encog.workbench.dialogs.training.neat.InputNEAT;
import org.encog.workbench.dialogs.training.neat.ProgressNEAT;
import org.encog.workbench.dialogs.training.outstar.InputOutstar;
import org.encog.workbench.dialogs.training.outstar.ProgressOutstar;
import org.encog.workbench.dialogs.training.resilient.InputResilient;
import org.encog.workbench.dialogs.training.resilient.ProgressResilient;
import org.encog.workbench.dialogs.training.scg.InputSCG;
import org.encog.workbench.dialogs.training.scg.ProgressSCG;
import org.encog.workbench.dialogs.training.som.InputSOM;
import org.encog.workbench.dialogs.training.som.ProgressSOM;
import org.encog.workbench.process.validate.ValidateTraining;

public class Training {
	
	private BasicNetwork network;
	private NeuralDataSet training;
	
	public void performAnneal() {
		final InputAnneal dialog = new InputAnneal(EncogWorkBench.getInstance()
				.getMainWindow());
		if (dialog.process()) {
			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised() ) {
				return;
			}		
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			final ProgressAnneal train = new ProgressAnneal(EncogWorkBench
					.getInstance().getMainWindow(), network, training, dialog
					.getMaxError().getValue(),
					dialog.getStartTemp().getValue(), dialog.getEndTemp()
							.getValue(), dialog.getCycles().getValue());

			train.setVisible(true);
		}
	}

	public void performBackpropagation() {
		final InputBackpropagation dialog = new InputBackpropagation(
				EncogWorkBench.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ProgressBackpropagation train = new ProgressBackpropagation(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getLearningRate().getValue(), dialog
							.getMomentum().getValue(), dialog.getMaxError()
							.getValue());

			train.setVisible(true);
		}
	}

	public void performGenetic() {
		final InputGenetic dialog = new InputGenetic(EncogWorkBench
				.getInstance().getMainWindow());
		if (dialog.process()) {
			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			final ProgressGenetic train = new ProgressGenetic(EncogWorkBench
					.getInstance().getMainWindow(), network, training, dialog
					.getMaxError().getValue(), dialog.getPopulationSize()
					.getValue(), dialog.getMutationPercent().getValue(), dialog
					.getPercentToMate().getValue());

			train.setVisible(true);
		}
	}

	public void performSOM() {
		final InputSOM dialog = new InputSOM(EncogWorkBench.getInstance()
				.getMainWindow());

		if (dialog.process()) {
			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsUnsupervised()) {
				return;
			}
			
			if( !validate.validateLogicType(SOMLogic.class)) {
				return;
			}

			final ProgressSOM train = new ProgressSOM(EncogWorkBench
					.getInstance().getMainWindow(), network, training, dialog
					.getMaxError().getValue(), dialog.getLearningRate()
					.getValue());

			train.setVisible(true);
		}

	}

	public void performResilient() {
		final InputResilient dialog = new InputResilient(EncogWorkBench
				.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ProgressResilient train = new ProgressResilient(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getInitialUpdate().getValue(), dialog
							.getMaxStep().getValue(), dialog.getMaxError()
							.getValue());

			train.setVisible(true);
		}

	}

	public void performManhattan() {
		final InputManhattan dialog = new InputManhattan(EncogWorkBench
				.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ProgressManhattan train = new ProgressManhattan(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getFixedDelta().getValue(), dialog
							.getMaxError().getValue());

			train.setVisible(true);
		}

	}
	
	public void performADALINE()
	{
		final InputAdaline dialog = new InputAdaline(
				EncogWorkBench.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ProgressAdaline train = new ProgressAdaline(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getLearningRate().getValue(), dialog.getMaxError()
							.getValue());

			train.setVisible(true);
		}
	}
	
	public void performInstar()
	{
		final InputInstar dialog = new InputInstar(
				EncogWorkBench.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ProgressInstar train = new ProgressInstar(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getLearningRate().getValue(), dialog.getMaxError()
							.getValue());

			train.setVisible(true);
		}
	}
	
	public void performOutstar()
	{
		final InputOutstar dialog = new InputOutstar(
				EncogWorkBench.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised() ) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			this.network = dialog.getNetwork();
			this.training = dialog.getTrainingSet();

			final ProgressOutstar train = new ProgressOutstar(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getLearningRate().getValue(), dialog.getMaxError()
							.getValue());

			train.setVisible(true);
		}
	}

	public void perform(Frame owner, BasicNetwork network) {
		ChooseTrainingMethodDialog dialog = new ChooseTrainingMethodDialog(
				owner, network);
		
		try
		{
			if(!EncogWorkBench.getInstance().getMainWindow().getSubwindows().checkTrainingOrNetworkOpen())
				return;
			
			if (dialog.process()) {
				switch (dialog.getType()) {
				case PropagationResilient:
					performResilient();
					break;
				case PropagationBack:
					performBackpropagation();
					break;
				case PropagationManhattan:
					performManhattan();
					break;
				case LevenbergMarquardt:
					performLMA();
					break;
				case Genetic:
					performGenetic();
					break;
				case Annealing:
					performAnneal();
					break;
				case SOM:
					performSOM();
					break;
				case ADALINE:
					performADALINE();
					break;
				case SCG:
					performSCG();
					break;
				case NEAT:
					performNEAT();
					break;
					
				}
			}
		}
		catch(Throwable t)
		{
			EncogWorkBench.displayError("Error While Training", t, this.network, this.training);
		}
	}
	
	public static void performSCG() {
		final InputSCG dialog = new InputSCG(EncogWorkBench
				.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			final BasicNetwork network = dialog.getNetwork();
			final NeuralDataSet training = dialog.getTrainingSet();

			final ProgressSCG train = new ProgressSCG(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getMaxError().getValue());

			train.setVisible(true);
		}

	}
	
	public static void performLMA() {
		final InputLMA dialog = new InputLMA(EncogWorkBench
				.getInstance().getMainWindow());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog
					.getNetwork(), (BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( !validate.validateFeedforwardOrSRN() ) {
				return;
			}

			final BasicNetwork network = dialog.getNetwork();
			final NeuralDataSet training = dialog.getTrainingSet();

			final ProgressLMA train = new ProgressLMA(
					EncogWorkBench.getInstance().getMainWindow(), network,
					training, dialog.getMaxError().getValue(),
					dialog.getUseBayesian().getValue());

			train.setVisible(true);
		}

	}
	
	public static void performNEAT() {
		final InputNEAT dialog = new InputNEAT(EncogWorkBench
				.getInstance().getMainWindow());
		dialog.getMaxError().setValue(EncogWorkBench.getInstance().getConfig().getDefaultError());
		if (dialog.process()) {
			final ValidateTraining validate = new ValidateTraining(dialog.getNetwork(), 
					(BasicNeuralDataSet) dialog.getTrainingSet());

			if (!validate.validateIsSupervised()) {
				return;
			}
			
			if( (!validate.validateNEAT())) {
				return;
			}

			final NeuralDataSet training = dialog.getTrainingSet();

			final ProgressNEAT train = new ProgressNEAT(
					EncogWorkBench.getInstance().getMainWindow(),
					training,
					dialog.getPopulation(),
					dialog.getNetwork(),
					dialog.getMaxError().getValue());

			train.setVisible(true);
		}

	}
}
