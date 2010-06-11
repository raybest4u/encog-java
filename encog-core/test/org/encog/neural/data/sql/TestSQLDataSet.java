/*
 * Encog(tm) Core v2.5 
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

package org.encog.neural.data.sql;

import java.sql.Connection;
import java.sql.Statement;

import junit.framework.TestCase;

import org.encog.neural.networks.XOR;
import org.encog.util.HSQLUtil;

public class TestSQLDataSet extends TestCase {
	
	public void testSQLDataSet() throws Exception
	{
		HSQLUtil.loadDriver();
		//DerbyUtil.cleanup();
		Connection conn = HSQLUtil.getConnection();
		
		conn.setAutoCommit(true);

		Statement s = conn.createStatement();

		// We create a table...
		StringBuilder sql = new StringBuilder();
		sql.append("CREATE TABLE \"XOR\" (");
		sql.append(" \"ID\" int GENERATED BY DEFAULT AS IDENTITY,");
		sql.append(" \"IN1\" int,");
		sql.append(" \"IN2\" int,");
		sql.append(" \"IDEAL1\" int");
		sql.append(" )");
		s.execute(sql.toString());
		
		s.execute("INSERT INTO xor(in1,in2,ideal1) VALUES(0,0,0)");
		s.execute("INSERT INTO xor(in1,in2,ideal1) VALUES(1,0,1)");
		s.execute("INSERT INTO xor(in1,in2,ideal1) VALUES(0,1,1)");
		s.execute("INSERT INTO xor(in1,in2,ideal1) VALUES(1,1,0)");
		
		SQLNeuralDataSet data = new SQLNeuralDataSet(
				"SELECT in1,in2,ideal1 FROM xor ORDER BY id",
				2,
				1, 
				HSQLUtil.DRIVER, 
				HSQLUtil.URL, 
				HSQLUtil.UID,
				HSQLUtil.PWD);
		
		XOR.testXORDataSet(data);
				
		HSQLUtil.shutdown();
		//DerbyUtil.cleanup();

	}
}
