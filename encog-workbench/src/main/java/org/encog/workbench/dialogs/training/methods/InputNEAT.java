/*
 * Encog(tm) Workbench v3.0
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
package org.encog.workbench.dialogs.training.methods;

import java.io.File;
import java.util.List;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.buffer.BufferedNeuralDataSet;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.dialogs.common.CheckField;
import org.encog.workbench.dialogs.common.ComboBoxField;
import org.encog.workbench.dialogs.common.DoubleField;
import org.encog.workbench.dialogs.common.EncogPropertiesDialog;
import org.encog.workbench.frames.document.tree.ProjectEGFile;
import org.encog.workbench.frames.document.tree.ProjectTraining;


public class InputNEAT extends EncogPropertiesDialog {

	private ComboBoxField comboTraining;
	private ComboBoxField comboPopulation;
	
	/**
	 * The serial id.
	 */
	private static final long serialVersionUID = 3506669325409959724L;

	/**
	 * All available training sets to display in the combo box.
	 */
	private List<ProjectTraining> trainingSets;
	
	/**
	 * All available networks to display in the combo box.
	 */
	private List<ProjectEGFile> populations;
	
	private final DoubleField maxError;
	
	private final CheckField loadToMemory;

	/**
	 * Construct the dialog box.
	 * @param owner The owner of the dialog box.
	 */
	public InputNEAT() {
		
		super(EncogWorkBench.getInstance().getMainWindow());
		findData();
		setTitle("Train NEAT Population");
		setSize(500,400);
		setLocation(200,200);
		addProperty(this.comboTraining = new ComboBoxField("training set","Training Set",true,this.trainingSets));
		addProperty(this.comboPopulation = new ComboBoxField("population","NEAT Population",true,this.populations));
		addProperty(this.loadToMemory = new CheckField("load to memory",
			"Load to Memory (better performance)"));
		addProperty(this.maxError = new DoubleField("max error",
				"Maximum Error Percent(0-100)", true, 0, 100));

		
		render();
		this.getLoadToMemory().setValue(true);
	}



	/**
	 * Obtain the data needed to fill in the network and training set
	 * combo boxes.
	 */
	private void findData() {
		this.trainingSets = EncogWorkBench.getInstance().getTrainingData();
		this.populations = EncogWorkBench.getInstance().getNEATPopulations();
	}

	/**
	 * @return The network that the user chose.
	 */
	public ProjectEGFile getPopulation() {
		if( this.comboPopulation.getSelectedValue()==null )
			return null;
		
		return ((ProjectEGFile)this.comboPopulation.getSelectedValue());
	}
	

	/**
	 * @return The training set that the user chose.
	 */
	public MLDataSet getTrainingSet() {
		if( this.comboTraining.getSelectedValue()==null )			
			return null;
		File file = ((ProjectTraining)this.comboTraining.getSelectedValue()).getFile();
		BufferedNeuralDataSet result = new BufferedNeuralDataSet(file);
		return result;
	}
	
	public ComboBoxField getComboTraining() {
		return this.comboTraining;
	}

	public ComboBoxField getComboPopulation() {
		return comboPopulation;
	}

	public List<ProjectTraining> getTrainingSets() {
		return trainingSets;
	}

	public List<ProjectEGFile> getPopulations() {
		return null;
	}


	public DoubleField getMaxError() {
		return maxError;
	}



	public CheckField getLoadToMemory() {
		return loadToMemory;
	}



}
