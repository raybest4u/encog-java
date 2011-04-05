package org.encog.plugins.logging;

import org.encog.plugin.EncogPluginBase;

public class EncogLoggingPlugin implements EncogPluginBase {

	@Override
	public int getPluginType() {
		return 1;
	}

	@Override
	public int getPluginFrameworkVersion() {
		return 1;
	}

	@Override
	public int getPluginVersion() {
		return 1;
	}

	@Override
	public String getPluginName() {
		return "hri-encog-logging";
	}

	@Override
	public String getPluginDescription() {
		return "Provides logging support for Encog.";
	}

	@Override
	public Object[] execute(String command, Object[] args) {
		return null;
	}

}
