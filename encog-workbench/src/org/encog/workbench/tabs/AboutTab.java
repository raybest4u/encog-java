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

import java.awt.Dimension;
import java.io.File;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

import org.encog.Encog;
import org.encog.util.HTMLReport;
import org.encog.workbench.EncogWorkBench;

public class AboutTab extends HTMLTab {
	
	/**
	 * JAR files currently in use.
	 */
	private final List<String> jars = new ArrayList<String>();

	public AboutTab() {
		super(null);
		
		setPreferredSize(new Dimension(800, 3000));
		
		final String path = System.getProperty("java.class.path");
		final StringTokenizer tok = new StringTokenizer(path, ""
				+ File.pathSeparatorChar);
		while (tok.hasMoreTokens()) {
			final String jarPath = tok.nextToken();
			final File jarFile = new File(jarPath);
			if (jarFile.isFile()) {
				try {
					final JarFile jarData = new JarFile(jarFile);
					final Manifest manifest = jarData.getManifest();
					this.jars.add(jarFile.getName() + " (" + jarFile + ")");

				} catch (final IOException e) {
					// ignore the JAR
				}
			}			
		}
		
		generate();
	}
	
	public void generate() {
		HTMLReport report = new HTMLReport();
		report.beginHTML();
		String title = "Encog Workbench v" + EncogWorkBench.VERSION;
		report.title(title);
		report.beginBody();
		report.h1(title);
		report.para("Encog Workbench is released under the Apache License.  For more information see the license file released with the Encog Workbench.");
		report.h3(EncogWorkBench.COPYRIGHT);
		
		report.beginTable();
		report.tablePair("Java Version", System.getProperty("java.version"));
		report.tablePair("Java 64/32-Bit", System.getProperty("sun.arch.data.model"));
		report.tablePair("Processor Count", ""+Runtime.getRuntime().availableProcessors());
		report.tablePair("OS Name/Version", ""+ByteOrder.nativeOrder().toString());
		report.tablePair("Encog Core Version", ""+Encog.VERSION);	
		report.endTable();

		report.h3("Active JAR Files");
		report.beginList();
		for (final String file : this.jars) {
			report.listItem(file);
		}		
		report.endList();
		report.endBody();
		report.endHTML();
		
		
		this.display(report.toString());
	}
	
	@Override
	public String getName() {
		return "About";
	}

}
