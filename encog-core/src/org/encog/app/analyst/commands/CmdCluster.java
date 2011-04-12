package org.encog.app.analyst.commands;

import java.io.File;

import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.evaluate.AnalystClusterCSV;
import org.encog.app.analyst.script.prop.ScriptProperties;
import org.encog.app.analyst.util.AnalystReportBridge;
import org.encog.util.csv.CSVFormat;

/**
 * This command is used to randomize the lines in a CSV file.
 *
 */
public class CmdCluster extends Cmd {

	public final static String COMMAND_NAME = "CLUSTER";	
	
	public CmdCluster(EncogAnalyst analyst) {
		super(analyst);
	}

	@Override
	public boolean executeCommand(String args) {
		// get filenames
		String sourceID = getProp().getPropertyString(ScriptProperties.CLUSTER_CONFIG_sourceFile);
		String targetID = getProp().getPropertyString(ScriptProperties.CLUSTER_CONFIG_targetFile);
		int clusters =  getProp().getPropertyInt(ScriptProperties.CLUSTER_CONFIG_clusters);
		String type = getProp().getPropertyString(ScriptProperties.CLUSTER_CONFIG_type);
		
		File sourceFile = this.getScript().resolveFilename(sourceID);
		File targetFile = this.getScript().resolveFilename(targetID);

		// get formats
		CSVFormat inputFormat = this.getScript().determineInputFormat(sourceID);
		CSVFormat outputFormat = this.getScript().determineOutputFormat(); 
			
		
		// mark generated
		getScript().markGenerated(targetID);

		// prepare to normalize
		AnalystClusterCSV cluster = new AnalystClusterCSV();		
		getAnalyst().setCurrentQuantTask(cluster);
		cluster.setReport(new AnalystReportBridge(getAnalyst()));
		boolean headers = this.getScript().expectInputHeaders(sourceID);
		cluster.analyze(getAnalyst(), sourceFile, headers, inputFormat);
		cluster.setOutputFormat(outputFormat);
		cluster.process(targetFile, clusters, this.getAnalyst());
		getAnalyst().setCurrentQuantTask(null);
		return cluster.shouldStop();
	}

	@Override
	public String getName() {
		return COMMAND_NAME;
	}

}
