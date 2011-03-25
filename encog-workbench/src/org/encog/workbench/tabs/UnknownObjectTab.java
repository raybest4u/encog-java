package org.encog.workbench.tabs;

import org.encog.util.HTMLReport;
import org.encog.workbench.frames.document.tree.ProjectEGFile;

public class UnknownObjectTab extends HTMLTab {

	public UnknownObjectTab(ProjectEGFile encogObject) {
		super(encogObject);
		HTMLReport report = new HTMLReport();
		report.beginHTML();
		String title = "Unknown Object Type";
		report.title(title);
		report.beginBody();
		report.h1(title);
		report.para("Unknown object: " + encogObject.getClass().getSimpleName());
		report.endBody();
		report.endHTML();
		this.display(report.toString());
		
	}

}
