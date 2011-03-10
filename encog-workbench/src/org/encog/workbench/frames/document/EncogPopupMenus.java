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
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;

import org.encog.persist.DirectoryEntry;
import org.encog.persist.EncogPersistedCollection;
import org.encog.util.file.FileUtil;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.frames.document.tree.ProjectDirectory;
import org.encog.workbench.frames.document.tree.ProjectEGItem;
import org.encog.workbench.frames.document.tree.ProjectFile;
import org.encog.workbench.frames.document.tree.ProjectItem;
import org.encog.workbench.process.CreateNewFile;
import org.encog.workbench.process.EncogAnalystWizard;
import org.encog.workbench.process.ImportExport;

public class EncogPopupMenus {

	private JPopupMenu popupNetwork;
	private JMenuItem popupNetworkDelete;

	private JMenuItem popupNetworkProperties;
	private JMenuItem popupNetworkOpen;
	private JMenuItem popupNetworkQuery;
	private JPopupMenu popupData;
	private JMenuItem popupDataDelete;

	private JMenuItem popupDataProperties;
	private JMenuItem popupDataOpen;
	private JMenuItem popupDataExport;

	private JPopupMenu popupGeneral;
	private JMenuItem popupGeneralOpen;
	private JMenuItem popupGeneralDelete;
	private JMenuItem popupGeneralProperties;
	private EncogDocumentFrame owner;

	private JPopupMenu popupFile;
	private JMenuItem popupFileDelete;
	private JMenuItem popupFileOpen;
	private JMenuItem popupFileOpenText;
	private JMenuItem popupFileRefresh;

	private JPopupMenu popupFileCSV;
	private JMenuItem popupFileCSVDelete;
	private JMenuItem popupFileCSVOpen;
	private JMenuItem popupFileCSVRefresh;
	private JMenuItem popupFileCSVExport;
	private JMenuItem popupFileCSVWizard;

	private JPopupMenu popupRefresh;
	private JMenuItem popupRefreshItem;
	
	private JPopupMenu popupRoot;
	private JMenuItem popupRootRefreshItem;
	private JMenuItem popupRootNewFile;

	public EncogPopupMenus(EncogDocumentFrame owner) {
		this.owner = owner;
	}

	void initPopup() {
		// build network popup menu
		this.popupNetwork = new JPopupMenu();
		this.popupNetworkDelete = owner.addItem(this.popupNetwork, "Delete",
				'd');
		this.popupNetworkOpen = owner.addItem(this.popupNetwork, "Open", 'o');
		this.popupNetworkProperties = owner.addItem(this.popupNetwork,
				"Properties", 'p');
		this.popupNetworkQuery = owner.addItem(this.popupNetwork, "Query", 'q');

		this.popupData = new JPopupMenu();
		this.popupDataDelete = owner.addItem(this.popupData, "Delete", 'd');
		this.popupDataOpen = owner.addItem(this.popupData, "Open", 'o');
		this.popupDataProperties = owner.addItem(this.popupData, "Properties",
				'p');
		this.popupDataExport = owner.addItem(this.popupData, "Export...", 'e');

		this.popupGeneral = new JPopupMenu();
		this.popupGeneralDelete = owner.addItem(this.popupGeneral, "Delete",
				'd');
		this.popupGeneralOpen = owner.addItem(this.popupGeneral, "Open", 'o');
		this.popupGeneralProperties = owner.addItem(this.popupGeneral,
				"Properties", 'p');

		this.popupFile = new JPopupMenu();
		this.popupFileOpen = owner.addItem(this.popupFile, "Open", 'o');
		this.popupFileOpenText = owner.addItem(this.popupFile, "Open as Text",
				't');
		this.popupFileDelete = owner.addItem(this.popupFile, "Delete", 'd');
		this.popupFileRefresh = owner.addItem(this.popupFile, "Refresh", 'r');

		this.popupRefresh = new JPopupMenu();
		this.popupRefreshItem = owner
				.addItem(this.popupRefresh, "Refresh", 'r');

		this.popupFileCSV = new JPopupMenu();
		this.popupFileCSVOpen = owner.addItem(this.popupFileCSV, "Open", 'o');
		this.popupFileCSVDelete = owner.addItem(this.popupFileCSV, "Delete",
				'd');
		this.popupFileCSVRefresh = owner.addItem(this.popupFileCSV, "Refresh",
				'r');
		this.popupFileCSVExport = owner.addItem(this.popupFileCSV,
				"Export to Training(EGB)", 'x');
		this.popupFileCSVWizard = owner.addItem(this.popupFileCSV,
				"Analyst Wizard...", 'w');
		
		this.popupRoot = new JPopupMenu();
		this.popupRootRefreshItem = owner
			.addItem(this.popupRoot, "Refresh", 'r');
		this.popupRootNewFile = owner
			.addItem(this.popupRoot, "New File", 'n');


	}

	public void actionPerformed(final ActionEvent event) {
		performPopupMenu(event.getSource());
	}

	public void performPopupMenu(final Object source) {

		if (source == this.popupFileRefresh || source == this.popupRefreshItem
				|| source == this.popupFileCSVRefresh) {
			EncogWorkBench.getInstance().getMainWindow().getTree().refresh();
		}

		boolean first = true;
		List<ProjectItem> list = this.owner.getTree().getSelectedValue();

		if (list == null)
			return;

		for (ProjectItem selected : list) {
			if (source == this.popupFileDelete) {
				if (first
						&& !EncogWorkBench
								.askQuestion("Warning",
										"Are you sure you want to delete these file(s)?")) {
					return;
				}
				first = false;
				if (selected instanceof ProjectFile) {
					((ProjectFile) selected).getFile().delete();
				}
				EncogWorkBench.getInstance().getMainWindow().getTree()
						.refresh();
			} else if (source == this.popupFileOpen
					|| source == this.popupFileCSVOpen) {
				if (selected instanceof ProjectFile) {
					EncogWorkBench.getInstance().getMainWindow()
							.openFile(((ProjectFile) selected).getFile());
				}
			} else if (source == this.popupFileOpenText) {
				if (selected instanceof ProjectFile) {
					EncogWorkBench.getInstance().getMainWindow()
							.openTextFile(((ProjectFile) selected).getFile());
				}
			} else if ((source == this.popupNetworkDelete)
					|| (source == this.popupDataDelete)
					|| (source == this.popupGeneralDelete)
					|| (source == this.popupFileCSVDelete)) {
				if (first
						&& !EncogWorkBench
								.askQuestion("Warning",
										"Are you sure you want to delete these object(s)?")) {
					return;
				}
				owner.getOperations().performObjectsDelete(selected);
			} else if (source == this.popupFileCSVExport) {
				String sourceFile = ((ProjectFile) selected).getFile()
						.toString();
				String targetFile = FileUtil.forceExtension(sourceFile, "egb");
				ImportExport.performExternal2Bin(new File(sourceFile),
						new File(targetFile), null);
			} else if (source == this.popupFileCSVWizard) {
				File sourceFile = ((ProjectFile) selected).getFile();
				EncogAnalystWizard.createEncogAnalyst(sourceFile);
			} else if (source == this.popupRootNewFile ) {
				try {
					CreateNewFile.performCreateFile();
				} catch (IOException e) {
					EncogWorkBench.displayError("Error", e);
				}
			}

			first = false;
		}
	}

	public void rightMouseClicked(final MouseEvent e, final Object item) {

		if (item instanceof DirectoryEntry) {
			DirectoryEntry entry = (DirectoryEntry) item;
			if (EncogPersistedCollection.TYPE_BASIC_NET.equals(entry.getType())) {
				this.popupNetwork.show(e.getComponent(), e.getX(), e.getY());
			} else if (EncogPersistedCollection.TYPE_BASIC_NET.equals(entry
					.getType())) {
				this.popupData.show(e.getComponent(), e.getX(), e.getY());
			} else {
				this.popupGeneral.show(e.getComponent(), e.getX(), e.getY());
			}
		} else if (item instanceof ProjectFile) {
			ProjectFile file = (ProjectFile) item;
			if (file.getExtension().equalsIgnoreCase("csv")) {
				this.popupFileCSV.show(e.getComponent(), e.getX(), e.getY());
			} else {
				this.popupFile.show(e.getComponent(), e.getX(), e.getY());
			}
		} else if (item instanceof ProjectEGItem) {
			this.popupGeneral.show(e.getComponent(), e.getX(), e.getY());
		} else if (item instanceof String) {
			this.popupRoot.show(e.getComponent(), e.getX(), e.getY());
		} else {
			this.popupRefresh.show(e.getComponent(), e.getX(), e.getY());
		}

	}

	public void performPopupDelete() {
		this.performPopupMenu(this.popupNetworkDelete);

	}

}
