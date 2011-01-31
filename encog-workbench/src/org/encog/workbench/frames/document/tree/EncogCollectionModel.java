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
package org.encog.workbench.frames.document.tree;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import javax.swing.event.TreeModelEvent;
import javax.swing.event.TreeModelListener;
import javax.swing.tree.TreeModel;
import javax.swing.tree.TreePath;

import org.encog.workbench.util.FileUtil;

public class EncogCollectionModel implements TreeModel {

	private String path;
	private List<ProjectItem> files = new ArrayList<ProjectItem>();
	private List<TreeModelListener> listeners = new ArrayList<TreeModelListener>();

	public EncogCollectionModel(String path) {
		invalidate(path);
	}

	public EncogCollectionModel() {
	}

	public Object getRoot() {
		return path;
	}

	public Object getChild(Object parent, int index) {
		if (parent == path) {
			return this.files.get(index);
		} else {
			if( parent instanceof ProjectEGFile ) {
				return ((ProjectEGFile)parent).getChildren()[index];
			}
			else {
				return null;
			}
		}
	}

	public int getChildCount(Object parent) {
		if (parent == path) {
			return this.files.size();
		} else {
			if( parent instanceof ProjectEGFile ) {
				return ((ProjectEGFile)parent).getChildren().length;
			}
			else {
				return 0;
			}
		}
	}

	public boolean isLeaf(Object node) {
		if (node == path) {
			return false;
		} else {
			if (node instanceof ProjectDirectory) {
				return false;
			} else if (node instanceof ProjectEGFile) {
				return false;
			} else if (node instanceof ProjectFile) {
				return true;
			}
			return true;
		}
	}

	public void valueForPathChanged(TreePath path, Object newValue) {
		// TODO Auto-generated method stub

	}

	public int getIndexOfChild(Object parent, Object child) {
		if (parent == path) {
			return this.files.indexOf(child);
		} else {
			return -1;
		}
	}

	public void addTreeModelListener(TreeModelListener l) {
		this.listeners.add(l);
	}

	public void removeTreeModelListener(TreeModelListener l) {
		this.listeners.remove(l);
	}

	public void invalidate(String path) {

		this.files.clear();

		if (path == null)
			return;

		this.path = path;

		// sort
		TreeSet<File> folderList = new TreeSet<File>();
		TreeSet<File> fileList = new TreeSet<File>();

		File file = new File(path);
		for (File entry : file.listFiles()) {
			if (!entry.isHidden()) {
				if (entry.isDirectory())
					folderList.add(entry);
				else
					fileList.add(entry);
			}

		}

		// build list
		this.files.clear();

		if (file.getParent() != null) {
			this.files.add(new ProjectParent(file.getParentFile()));
		}

		for (File entry : folderList) {
			this.files.add(new ProjectDirectory(entry));
		}

		for (File entry : fileList) {
			if (FileUtil.getFileExt(entry).equalsIgnoreCase("eg")) {
				try {					
					this.files.add(new ProjectEGFile(entry));
				} catch (Throwable t) {
					this.files.add(new ProjectFile(entry));
				}
			} else {
				this.files.add(new ProjectFile(entry));
			}
		}

		// notify
		Object[] p = new Object[1];
		p[0] = this.path;
		TreeModelEvent e = new TreeModelEvent(this, p);
		for (TreeModelListener l : this.listeners) {
			l.treeStructureChanged(e);
		}
	}

	public void invalidate() {
		invalidate(this.path);
	}

	public String getPath() {
		return this.path;
	}

	public String[] listEGFiles() {
		List<String> files = new ArrayList<String>();
		
		for( ProjectItem item: this.files )
		{
			if( item instanceof ProjectEGFile )
			{
				files.add( ((ProjectEGFile)item).getFile().getName() );
			}
		}
		
		String[] result = new String[files.size()];
		files.toArray(result);
		
		return result;
	}

	public ProjectFile findTreeFile(String filename) {

		for( ProjectItem item: this.files )
		{
			if( item instanceof ProjectFile )
			{
				if( ((ProjectFile)item).getFile().getName().equalsIgnoreCase(filename) )
				{
					return (ProjectFile)item;
				}
			}
		}
				
		return null;
	}

}
