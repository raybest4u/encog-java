package org.encog.app.analyst.commands;

import java.io.File;

import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.script.prop.ScriptProperties;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

/**
 * This command is used to generate the binary EGB file from a CSV file. The
 * resulting file can be used for training.
 * 
 */
public class CmdGenerate extends Cmd {

	public final static String COMMAND_NAME = "GENERATE";

	public CmdGenerate(EncogAnalyst analyst) {
		super(analyst);
	}

	@Override
	public boolean executeCommand() {
		// get filenames
		String sourceID = getProp().getPropertyString(
				ScriptProperties.GENERATE_CONFIG_sourceFile);
		String targetID = getProp().getPropertyString(
				ScriptProperties.GENERATE_CONFIG_targetFile);
		CSVFormat format = this.getAnalyst().getScript()
				.determineInputFormat(sourceID);

		File sourceFile = getScript().resolveFilename(sourceID);
		File targetFile = getScript().resolveFilename(targetID);

		// mark generated
		getScript().markGenerated(targetID);

		// int input =
		// getProp().getPropertyInt(ScriptProperties.GENERATE_CONFIG_input);
		// int ideal =
		// getProp().getPropertyInt(ScriptProperties.GENERATE_CONFIG_ideal);

		boolean headers = getScript().expectInputHeaders(sourceID);
		int[] input = this.getAnalyst().determineInputFields();
		int[] ideal = this.getAnalyst().determineIdealFields();

		EncogUtility.convertCSV2Binary(sourceFile, format, targetFile, input,
				ideal, headers);
		return false;
	}

	@Override
	public String getName() {
		return COMMAND_NAME;
	}

}
