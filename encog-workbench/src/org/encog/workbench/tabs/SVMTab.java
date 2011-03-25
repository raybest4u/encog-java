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
package org.encog.workbench.tabs;

import org.encog.ml.svm.SVM;
import org.encog.util.HTMLReport;
import org.encog.workbench.frames.document.tree.ProjectEGFile;

public class SVMTab extends HTMLTab {

	private SVM network;
	
	public SVMTab(ProjectEGFile encogObject) {
		super(encogObject);
		this.network = (SVM)encogObject.getObject();
		
		HTMLReport report = new HTMLReport();
		String title = "Support Vector Machine (SVM)";
		report.beginHTML();
		report.title(title);
		report.beginBody();
		report.h1(title);
		report.beginTable();
		SVM svm = (SVM)encogObject.getObject();
		report.tablePair("Input Count",""+svm.getInputCount());
		report.tablePair("SVM Type",svm.getSVMType().toString());
		report.tablePair("Kernel Type",svm.getKernelType().toString());
		report.endTable();
		report.endBody();
		report.endHTML();
		
		this.display(report.toString());
	}
	

}
