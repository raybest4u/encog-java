package org.encog.app.analyst.script;

import java.io.InputStream;
import java.io.OutputStream;

import org.encog.app.analyst.script.normalize.AnalystNormalize;
import org.encog.app.analyst.script.normalize.NormalizedField;

public class AnalystScript {

	private final EncogAnalystConfig config = new EncogAnalystConfig();
	private DataField[] fields;
	private final AnalystNormalize normalize = new AnalystNormalize();

	/**
	 * @return the config
	 */
	public EncogAnalystConfig getConfig() {
		return config;
	}

	/**
	 * @return the fields
	 */
	public DataField[] getFields() {
		return fields;
	}

	/**
	 * @param fields the fields to set
	 */
	public void setFields(DataField[] fields) {
		this.fields = fields;
	}


	public void save(OutputStream stream) {
		ScriptSave s = new ScriptSave(this);
		s.save(stream);
	}
	
	public void load(InputStream stream) {
		ScriptLoad s = new ScriptLoad(this);
		s.load(stream);
	}

	/**
	 * @return the normalize
	 */
	public AnalystNormalize getNormalize() {
		return normalize;
	}
		
	
}
