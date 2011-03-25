package org.encog.workbench.frames.document.tree;

import java.io.File;

import org.encog.persist.EncogDirectoryPersistence;
import org.encog.workbench.EncogWorkBench;

public class ProjectEGFile extends ProjectFile {

	private Object obj;
	private String encogType;

	public ProjectEGFile(File file) {
		super(file);

		try {
			this.encogType = EncogWorkBench.getInstance().getProject()
					.getEncogType(file.getName());
		} catch (Throwable t) {
			this.encogType = "Unknown";
		}
	}

	public String toString() {
		return this.getFile().getName() + " (" + encogType + ")";
	}

	public Object getObject() {
		return EncogWorkBench.getInstance().getProject().loadFromDirectory(getName());
	}
	
	public void save() {
		if( this.getObject()!=null ) {
			EncogWorkBench.getInstance().getProject().saveToDirectory(this.getName(), this.getObject());
		}
	}

	/**
	 * @return the encogType
	 */
	public String getEncogType() {
		return encogType;
	}

}
