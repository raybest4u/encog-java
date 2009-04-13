package org.encog.workbench.config;

import org.encog.EncogError;
import org.encog.persist.EncogPersistedObject;
import org.encog.persist.Persistor;

public class EncogWorkBenchConfig implements EncogPersistedObject {
	
	private String databaseConnectionString = "";
	private String databaseDriver = "";
	private String databaseUserID = "";
	private String databasePassword = "";
	private String databaseDialect = "";
	public String getDatabaseConnectionString() {
		return databaseConnectionString;
	}
	public void setDatabaseConnectionString(String databaseConnectionString) {
		this.databaseConnectionString = databaseConnectionString;
	}
	public String getDatabaseDriver() {
		return databaseDriver;
	}
	public void setDatabaseDriver(String databaseDriver) {
		this.databaseDriver = databaseDriver;
	}
	public String getDatabaseUserID() {
		return databaseUserID;
	}
	public void setDatabaseUserID(String databaseUserID) {
		this.databaseUserID = databaseUserID;
	}
	public String getDatabasePassword() {
		return databasePassword;
	}
	public void setDatabasePassword(String databasePassword) {
		this.databasePassword = databasePassword;
	}

	public Persistor createPersistor() {
		return null;
	}

	public String getDescription() {
		return null;
	}

	public String getName() {
		return "config";
	}

	public void setDescription(String theDescription) {
	}
	
	public void setName(String theName) {
	}
	public String getDatabaseDialect() {
		return databaseDialect;
	}
	public void setDatabaseDialect(String databaseDialect) {
		this.databaseDialect = databaseDialect;
	}
	
	public Object clone()
	{
		throw new EncogError("Clone not supported");
	}
	
	
}
