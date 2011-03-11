/*
 * Encog(tm) Workbench v2.6 
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2010 Heaton Research, Inc.
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
package org.encog.workbench.dialogs.wizard.analyst;

import java.awt.Frame;
import java.util.ArrayList;
import java.util.List;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.AnalystGoal;
import org.encog.app.analyst.wizard.NormalizeRange;
import org.encog.app.analyst.wizard.WizardMethodType;
import org.encog.workbench.dialogs.common.CheckField;
import org.encog.workbench.dialogs.common.ComboBoxField;
import org.encog.workbench.dialogs.common.EncogPropertiesDialog;
import org.encog.workbench.dialogs.common.FileField;
import org.encog.workbench.dialogs.common.IntegerField;
import org.encog.workbench.dialogs.common.TextField;
import org.encog.workbench.frames.document.EncogDocumentFrame;

public class AnalystWizardDialog extends EncogPropertiesDialog {
	
	private final FileField rawFile;
	private final ComboBoxField method;
	private final ComboBoxField format;
	private final ComboBoxField range;
	private final ComboBoxField goal;
	private final TextField targetField;
	private final CheckField headers;
	private final IntegerField lagCount;
	private final IntegerField leadCount;
	private final CheckField includeTarget;
	private final CheckField normalize;
	private final CheckField segregate;
	private final CheckField randomize;
	
	private final List<String> methods = new ArrayList<String>();
	
	public AnalystWizardDialog(Frame owner) {
		super(owner);
		
		List<String> list = new ArrayList<String>();
		list.add("CSV");
		
		List<String> csvFormat = new ArrayList<String>();
		csvFormat.add("Decimal Point (USA/English) & Comma Separator");
		csvFormat.add("Decimal Point (USA/English) & Space Separator");
		csvFormat.add("Decimal Point (USA/English) & Semicolon Separator");
		csvFormat.add("Decimal Comma (Non-USA/English) & Space Separator");
		csvFormat.add("Decimal Comma (Non-USA/English) & Semicolon Separator");
		
		List<String> goalList = new ArrayList<String>();
		goalList.add("Classification");
		goalList.add("Regression");
		
		List<String> rangeList = new ArrayList<String>();
		rangeList.add("-1 to 1");
		rangeList.add("0 to 1");

		
		methods.add("Feedforward Network");
		methods.add("RBF Network");
		methods.add("Support Vector Machine");
		methods.add("PNN/GRNN Network");
		
		this.setSize(640, 300);
		this.setTitle("Setup Encog Analyst Wizard");
		
		beginTab("General");
		addProperty(this.rawFile = new FileField("source file","Source CSV File(*.csv)",true,false,EncogDocumentFrame.CSV_FILTER));
		addProperty(this.format = new ComboBoxField("format", "File Format", true, csvFormat));
		addProperty(this.method = new ComboBoxField("method", "Machine Learning", true, methods));
		addProperty(this.goal = new ComboBoxField("goal", "Goal", true, goalList));
		addProperty(this.targetField = new TextField("target field", "Target Field(blank for auto)", false));
		addProperty(this.headers = new CheckField("headers","CSV File Headers"));
		addProperty(this.range = new ComboBoxField("normalization range", "Normalization Range", true, rangeList));
		beginTab("Time Series");
		addProperty(this.lagCount = new IntegerField("lag count","Lag Count",true,0,1000));
		addProperty(this.leadCount = new IntegerField("lead count","Lead Count",true,0,1000));
		addProperty(this.includeTarget = new CheckField("include target","Include Target in Input"));
		beginTab("Tasks");
		addProperty(this.normalize = new CheckField("normalize","Normalize"));
		addProperty(this.randomize = new CheckField("randomize","Randomize"));
		addProperty(this.segregate = new CheckField("segregate","Segregate"));
		
		
		render();
		
		this.lagCount.setValue(0);
		this.leadCount.setValue(0);
		
		this.randomize.setValue(true);
		this.segregate.setValue(true);
		this.normalize.setValue(true);
	
	}

	/**
	 * @return the rawFile
	 */
	public FileField getRawFile() {
		return rawFile;
	}

	/**
	 * @return the headers
	 */
	public CheckField getHeaders() {
		return headers;
	}

	
	public WizardMethodType getMethodType()
	{
		switch(this.method.getSelectedIndex()) {
			case 0:
				return WizardMethodType.FeedForward;
			case 1:
				return WizardMethodType.RBF;
			case 2:
				return WizardMethodType.SVM;
			case 3:
				return WizardMethodType.PNN;
			default:
				return null;
		}
	}
	
	public AnalystFileFormat getFormat() {
		int idx = this.format.getSelectedIndex(); 
		switch( idx ) {
			case 0:
				return AnalystFileFormat.DECPNT_COMMA;
			case 1:
				return AnalystFileFormat.DECPNT_SPACE;
			case 2:
				return AnalystFileFormat.DECPNT_SEMI;
			case 3:
				return AnalystFileFormat.DECCOMMA_SPACE;
			case 4:
				return AnalystFileFormat.DECCOMMA_SEMI;
			default:
				return null;
		}
	}
	
	public AnalystGoal getGoal() {
		int idx = this.goal.getSelectedIndex();
		switch(idx) {
			case 0:
				return AnalystGoal.Classification;
			case 1:
				return AnalystGoal.Regression;
			default:
				return null;
		}
	}
	
	public NormalizeRange getRange() {
		int idx = this.range.getSelectedIndex();
		switch(idx) {
			case 0:
				return NormalizeRange.NegOne2One;
			case 1:
				return NormalizeRange.Zero2One;
			default:
				return null;
		}
	}
	
	public String getTargetField() {
		return this.targetField.getValue();
	}

	/**
	 * @return the lagCount
	 */
	public IntegerField getLagCount() {
		return lagCount;
	}

	/**
	 * @return the leadCount
	 */
	public IntegerField getLeadCount() {
		return leadCount;
	}

	/**
	 * @return the includeTarget
	 */
	public CheckField getIncludeTarget() {
		return includeTarget;
	}

	/**
	 * @return the segregate
	 */
	public CheckField getSegregate() {
		return segregate;
	}

	/**
	 * @return the randomize
	 */
	public CheckField getRandomize() {
		return randomize;
	}

	/**
	 * @return the normalize
	 */
	public CheckField getNormalize() {
		return normalize;
	}	
	
	
}
