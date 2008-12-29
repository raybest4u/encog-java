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
package org.encog.workbench.frames.document;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.MouseEvent;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;

import org.encog.EncogError;
import org.encog.bot.spider.SpiderOptions;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.PropertyData;
import org.encog.neural.data.TextData;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.Network;
import org.encog.neural.networks.layers.FeedforwardLayer;
import org.encog.neural.persist.EncogPersistedCollection;
import org.encog.neural.persist.EncogPersistedObject;
import org.encog.parse.ParseTemplate;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.dialogs.CreateDataSet;
import org.encog.workbench.dialogs.EditEncogObjectProperties;
import org.encog.workbench.dialogs.EditSpiderOptions;
import org.encog.workbench.dialogs.about.AboutEncog;
import org.encog.workbench.dialogs.select.SelectDialog;
import org.encog.workbench.dialogs.select.SelectItem;
import org.encog.workbench.frames.BrowserFrame;
import org.encog.workbench.frames.Conversation;
import org.encog.workbench.frames.EncogListFrame;
import org.encog.workbench.frames.NetworkFrame;
import org.encog.workbench.frames.NetworkQueryFrame;
import org.encog.workbench.frames.TrainingDataFrame;
import org.encog.workbench.frames.manager.EncogCommonFrame;
import org.encog.workbench.frames.render.EncogItemRenderer;
import org.encog.workbench.frames.visualize.NetworkVisualizeFrame;
import org.encog.workbench.models.EncogListModel;
import org.encog.workbench.process.ImportExport;
import org.encog.workbench.process.Training;
import org.encog.workbench.process.generate.CodeGeneration;
import org.encog.workbench.util.ExtensionFilter;
import org.encog.workbench.util.ImportExportUtility;
import org.encog.workbench.util.NeuralConst;

public class EncogDocumentFrame extends EncogListFrame {

	private EncogDocumentOperations operations;
	private EncogMenus menus;
	private EncogPopupMenus popupMenus;

	public static final ExtensionFilter ENCOG_FILTER = new ExtensionFilter(
			"Encog Files", ".eg");
	public static final ExtensionFilter CSV_FILTER = new ExtensionFilter(
			"CSV Files", ".csv");
	public static final String WINDOW_TITLE = "Encog Workbench 1.0";
	/**
	 * 
	 */
	private static final long serialVersionUID = -4161616483326975155L;
	



	private final EncogListModel encogListModel;

	public EncogDocumentFrame() {
		this.setSize(640, 480);
		
		this.operations = new EncogDocumentOperations(this);
		this.menus = new EncogMenus(this);
		this.popupMenus = new EncogPopupMenus(this);

		addWindowListener(this);
		EncogWorkBench.getInstance().setCurrentFile(
				new EncogPersistedCollection());
		this.encogListModel = new EncogListModel(EncogWorkBench.getInstance()
				.getCurrentFile());
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		this.menus.initMenuBar();
		initContents();
		this.popupMenus.initPopup();
	}

	public void actionPerformed(final ActionEvent event) {
		this.menus.actionPerformed(event);
		this.popupMenus.actionPerformed(event);
	}

	

	

	private void initContents() {
		// setup the contents list
		this.contents = new JList(this.encogListModel);
		this.contents.setCellRenderer(new EncogItemRenderer());
		this.contents.setFixedCellHeight(72);
		this.contents.addMouseListener(this);

		final JScrollPane scrollPane = new JScrollPane(this.contents);

		getContentPane().add(scrollPane);
		redraw();
	}



	

	public void redraw() {

		// set the title properly
		if (EncogWorkBench.getInstance().getCurrentFileName() == null) {
			setTitle(EncogDocumentFrame.WINDOW_TITLE + " : Untitled");
		} else {
			setTitle(EncogDocumentFrame.WINDOW_TITLE + " : "
					+ EncogWorkBench.getInstance().getCurrentFileName());
		}

		// redraw the list
		this.encogListModel.invalidate();
	}

	public void rightMouseClicked(final MouseEvent e, final Object item) {
		this.popupMenus.rightMouseClicked(e, item);
	}

	public void windowClosed(final WindowEvent e) {
		System.exit(0);

	}

	public void windowOpened(final WindowEvent e) {
	}

	/**
	 * @return the operations
	 */
	public EncogDocumentOperations getOperations() {
		return operations;
	}

	/**
	 * @return the menus
	 */
	public EncogMenus getMenus() {
		return menus;
	}

	/**
	 * @return the popupMenus
	 */
	public EncogPopupMenus getPopupMenus() {
		return popupMenus;
	}

	/**
	 * @return the contents
	 */
	public JList getContents() {
		return contents;
	}

	@Override
	protected void openItem(Object item) {
		this.operations.openItem(item);
		
	}
	
	
}
