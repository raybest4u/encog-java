/*
 * Encog(tm) Artificial Intelligence Framework v2.3
 * Java Version, Unit Tests
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2010 by Heaton Research Inc.
 * 
 * Released under the LGPL.
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
 * 
 * Encog and Heaton Research are Trademarks of Heaton Research, Inc.
 * For information on Heaton Research trademarks, visit:
 * 
 * http://www.heatonresearch.com/copyright.html
 */

package org.encog.util;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utilities for using the Hypersonic SQL (HSQL) engine. Encog uses this SQL
 * database for in-memory spidering, as well as unit testing.
 * 
 * @author jheaton
 * 
 */
public final class HSQLUtil {

	/**
	 * The driver to use for the memory database.
	 */
	public static final String DRIVER = "org.hsqldb.jdbcDriver";
	
	/**
	 * The URL to use for the memory database.
	 */
	public static final String URL = "jdbc:hsqldb:mem:encog";
	
	/**
	 * The user id to use for the memory database.
	 */
	public static final String UID = "sa";
	
	/**
	 * The password to use for the memory database.
	 */
	public static final String PWD = "";
	
	/**
	 * The dialect to use for the memory database.
	 */
	public static final String DIALECT = "org.hibernate.dialect.HSQLDialect";

	/**
	 * @return A connection to the memory database.
	 * @throws SQLException An SQL error.
	 */
	public static Connection getConnection() throws SQLException {
		final Properties props = new Properties();
		props.put("user", HSQLUtil.UID);
		props.put("password", HSQLUtil.PWD);

		return DriverManager.getConnection(HSQLUtil.URL, props);
	}



	/**
	 * Load the driver for the memory database.
	 * @throws InstantiationException Database error.
	 * @throws IllegalAccessException Database error.
	 * @throws ClassNotFoundException Database error.
	 */
	public static void loadDriver() throws InstantiationException,
			IllegalAccessException, ClassNotFoundException {
		Class.forName("org.hsqldb.jdbcDriver");
	}

	/**
	 * Shutdown the database. Not currently implemented.
	 */
	public static void shutdown() {

	}

	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	private final Logger logger = LoggerFactory.getLogger(this.getClass());
	
	/**
	 * Private constructor.
	 */
	private HSQLUtil() {		
	}
}
