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
package org.encog.workbench.frames.document;

import java.awt.event.ActionEvent;

import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JTabbedPane;

import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.process.CreateNewFile;
import org.encog.workbench.process.ImportExport;
import org.encog.workbench.tabs.EncogCommonTab;
import org.encog.workbench.tabs.TextDataTab;
import org.encog.workbench.tabs.ValidationChart;

public class EncogMenus {
	public static final String FILE_CHANGE_DIR = "Change Directory/Open Project";
	public static final String FILE_NEW_PROJECT = "New Project...";
	public static final String FILE_NEW_FILE = "New File...";
	public static final String FILE_QUIT = "Quit...";

	public static final String EDIT_CUT = "Cut";
	public static final String EDIT_COPY = "Copy";
	public static final String EDIT_PASTE = "Paste";
	public static final String EDIT_CONFIG = "Config...";

	public static final String OBJECTS_CREATE = "Create Object...";
	public static final String OBJECTS_DELETE = "Delete Object...";

	//public static final String TOOLS_CLOUD = "Connect to an Encog Cloud...";
	public static final String TOOLS_TRAIN = "Train...";
	public static final String TOOLS_GENERATE_TRAINING = "Generate Training Data...";
	public static final String TOOLS_EVALUATE = "Evaluate Network...";
	public static final String TOOLS_BENCHMARK = "Benchmark Encog...";
	public static final String TOOLS_BROWSE = "Browse Web Data...";
	public static final String TOOLS_VALIDATION_CHART = "Validation Chart...";
	public static final String TOOLS_BIN2EXTERNAL = "Convert Encog Binary to Other File...";
	public static final String TOOLS_EXTERNAL2BIN = "Convert Other File to Encog Binary...";
	public static final String TOOLS_RBF = "Chart RBF...";
	
		
	public static final String HELP_ABOUT = "About Encog Workbench...";

	private JMenuBar menuBar;
	private JMenu menuFile;
	private JMenu menuEdit;
	private JMenu menuObjects;
	private JMenu menuHelp;
	private JMenu menuTools;
	
	private JMenuItem menuFileNew;
	private JMenuItem menuFileQuit;

	private JMenuItem menuEditCut;
	private JMenuItem menuEditCopy;
	private JMenuItem menuEditPaste;
	private JMenuItem menuEditConfig;

	private JMenuItem menuObjectsCreate;
	private JMenuItem menuObjectsDelete;

	private JMenuItem menuToolsTrain;
	private JMenuItem menuToolsGenerate;
	private JMenuItem menuToolsEvaluate;
	private JMenuItem menuToolsBenchmark;
	private JMenuItem menuToolsBrowse;
	private JMenuItem menuToolsValidation;
	private JMenuItem menuToolsBin2Ext;
	private JMenuItem menuToolsExt2Bin;
	private JMenuItem menuToolsRBF;	
	
	private JMenuItem menuHelpAbout;

	private EncogDocumentFrame owner;

	public EncogMenus(EncogDocumentFrame owner) {
		this.owner = owner;
	}

	void initMenuBar() {
		// menu bar
		this.menuBar = new JMenuBar();
		this.menuFile = new JMenu("File");
		this.menuFileNew = this.menuFile.add(owner
				.addItem(this.menuFile, EncogMenus.FILE_NEW_PROJECT, 'n'));
		this.menuFileNew = this.menuFile.add(owner
				.addItem(this.menuFile, EncogMenus.FILE_CHANGE_DIR, 'c'));
		this.menuFileNew = this.menuFile.add(owner
				.addItem(this.menuFile, EncogMenus.FILE_NEW_FILE, 'f'));
		this.menuFile.addSeparator();
		this.menuFileQuit = this.menuFile.add(owner.addItem(this.menuFile, EncogMenus.FILE_QUIT,
				'q'));
		this.menuFile.addActionListener(this.owner);
		this.menuBar.add(this.menuFile);

		this.menuEdit = new JMenu("Edit");
		this.menuEditCut = this.menuEdit.add(owner
				.addItem(this.menuEdit, EncogMenus.EDIT_CUT, 'x'));
		this.menuEditCopy = this.menuEdit.add(owner.addItem(this.menuEdit, EncogMenus.EDIT_COPY,
				'c'));
		this.menuEditPaste = this.menuEdit.add(owner.addItem(this.menuEdit, EncogMenus.EDIT_PASTE,
				'v'));
		this.menuEdit.addSeparator();
		this.menuEditConfig = this.menuEdit.add(owner.addItem(this.menuEdit,
				EncogMenus.EDIT_CONFIG, 'f'));
		this.menuBar.add(this.menuEdit);

		this.menuObjects = new JMenu("Objects");
		this.menuObjectsCreate = this.menuObjects.add(owner.addItem(this.menuObjects,
				EncogMenus.OBJECTS_CREATE, 'c'));
		this.menuObjectsDelete = this.menuObjects.add(owner.addItem(this.menuObjects,
				EncogMenus.OBJECTS_DELETE, 'd'));
		this.menuBar.add(this.menuObjects);

		this.menuTools = new JMenu("Tools");
		//this.menuToolsCloud = owner.addItem(this.menuTools, EncogMenus.TOOLS_CLOUD, 'c');
		this.menuToolsGenerate = owner.addItem(this.menuTools, EncogMenus.TOOLS_GENERATE_TRAINING, 'g');
		this.menuToolsTrain = owner.addItem(this.menuTools, EncogMenus.TOOLS_TRAIN, 't');
		this.menuToolsBenchmark = owner.addItem(this.menuTools, EncogMenus.TOOLS_BENCHMARK, 'k');
		this.menuToolsEvaluate = owner.addItem(this.menuTools, EncogMenus.TOOLS_EVALUATE, 'e');
		this.menuToolsBrowse = owner.addItem(this.menuTools, EncogMenus.TOOLS_BROWSE, 'b');
		this.menuToolsValidation = owner.addItem(this.menuTools, EncogMenus.TOOLS_VALIDATION_CHART, 'v');
		this.menuToolsBin2Ext = owner.addItem(this.menuTools, EncogMenus.TOOLS_BIN2EXTERNAL, 'x');
		this.menuToolsExt2Bin = owner.addItem(this.menuTools, EncogMenus.TOOLS_EXTERNAL2BIN, 'y');
		this.menuToolsRBF = owner.addItem(this.menuTools, EncogMenus.TOOLS_RBF, 'r');
		this.menuBar.add(this.menuTools);		

		this.menuHelp = new JMenu("Help");
		this.menuHelpAbout = this.menuHelp.add(owner.addItem(this.menuHelp, EncogMenus.HELP_ABOUT,
				'a'));
		this.menuBar.add(this.menuHelp);

		owner.setJMenuBar(this.menuBar);
	}
	
	public void updateMenus()
	{
		boolean modal = this.owner.isModalTabOpen();
		boolean supportsClipboard = false;
		boolean documentOpen = EncogWorkBench.getInstance().getCurrentFileName()!=null;
		
		JTabbedPane tabs = EncogWorkBench.getInstance().getMainWindow().getDocumentTabs();

		
		EncogCommonTab currentTab = (EncogCommonTab)tabs.getSelectedComponent();
		
		if( currentTab instanceof TextDataTab )
		{
			supportsClipboard = true;
		}
		
		this.menuFileNew.setEnabled(!modal && documentOpen);	
		this.menuFileQuit.setEnabled(true);

		this.menuEditCut.setEnabled(!modal && supportsClipboard && documentOpen);
		this.menuEditCopy.setEnabled(!modal && supportsClipboard && documentOpen);
		this.menuEditPaste.setEnabled(!modal && supportsClipboard && documentOpen);
		this.menuEditConfig.setEnabled(!modal);

		this.menuObjectsCreate.setEnabled(!modal && documentOpen);
		this.menuObjectsDelete.setEnabled(!modal && documentOpen);

		//this.menuToolsCloud.setEnabled(!modal);
		this.menuToolsTrain.setEnabled(!modal && documentOpen);
		this.menuToolsGenerate.setEnabled(!modal && documentOpen);
		this.menuToolsEvaluate.setEnabled(!modal && documentOpen);
		this.menuToolsBenchmark.setEnabled(!modal);
		this.menuToolsBrowse.setEnabled(!modal);
		this.menuToolsValidation.setEnabled(!modal && documentOpen);

		this.menuHelpAbout.setEnabled(!modal);
	}

	public void actionPerformed(final ActionEvent event) {
		
		try {
		EncogWorkBench.getInstance().getMainWindow().endWait();
		if (event.getActionCommand().equals(EncogMenus.FILE_CHANGE_DIR)) {
			owner.getOperations().performFileChooseDirectory();
		} else if (event.getActionCommand().equals(EncogMenus.FILE_NEW_PROJECT)) {
			owner.getOperations().performFileNewProject();
		} else if (event.getActionCommand().equals(EncogMenus.FILE_NEW_FILE)) {
			CreateNewFile.performCreateFile();
		} else if (event.getActionCommand().equals(EncogMenus.FILE_QUIT)) {
			owner.getOperations().performQuit();
		} else if (event.getActionCommand().equals(EncogMenus.OBJECTS_CREATE)) {
			owner.getOperations().performObjectsCreate();
		} else if (event.getActionCommand().equals(EncogMenus.OBJECTS_DELETE)) {
			owner.getPopupMenus().performPopupDelete();
		} else if (event.getActionCommand().equals(EncogMenus.EDIT_CUT)) {
			owner.getOperations().performEditCut();
		} else if (event.getActionCommand().equals(EncogMenus.EDIT_COPY)) {
			owner.getOperations().performEditCopy();
		} else if (event.getActionCommand().equals(EncogMenus.EDIT_PASTE)) {
			owner.getOperations().performEditPaste();
		} else if (event.getActionCommand().equals(EncogMenus.EDIT_CONFIG)) {
			owner.getOperations().performEditConfig();
		} else if (event.getActionCommand().equals(EncogMenus.HELP_ABOUT)) {
			owner.getOperations().performHelpAbout();
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_BENCHMARK)) {
			owner.getOperations().performBenchmark();
		} /*else if (event.getActionCommand().equals(EncogMenus.TOOLS_CLOUD)) {
			owner.getOperations().performCloudLogin();
		}*/ else if (event.getActionCommand().equals(EncogMenus.TOOLS_BROWSE)) {
			owner.getOperations().performBrowse();
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_EVALUATE)) {
			owner.getOperations().performEvaluate();
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_TRAIN)) {
			owner.getOperations().performTrain();
		} else if (event.getActionCommand().equals(
				EncogMenus.TOOLS_VALIDATION_CHART)) {
			ValidationChart check = new ValidationChart();
			check.perform(EncogWorkBench.getInstance().getMainWindow());
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_BIN2EXTERNAL)) {
			ImportExport.performBin2External();
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_EXTERNAL2BIN)) {
			ImportExport.performExternal2Bin(null,null,null);
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_RBF)) {
			owner.getOperations().performRBF();
		} else if (event.getActionCommand().equals(EncogMenus.TOOLS_GENERATE_TRAINING)) {
			owner.getOperations().performCreateTrainingData("");
		}
	
		
		}
		catch(Throwable t)
		{
			EncogWorkBench.displayError("Error", t);
		}
	}
}
