/*
 * Encog(tm) Core Unit Tests v3.0 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
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
package org.encog.persist;

import java.io.File;
import java.io.IOException;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.encog.neural.art.ART1;
import org.encog.neural.som.SOM;
import org.encog.util.TempDir;
import org.encog.util.obj.SerializeObject;

public class TestPersistSOM extends TestCase {
	
	public final TempDir TEMP_DIR = new TempDir();
	public final File EG_FILENAME = TEMP_DIR.createFile("encogtest.eg");
	public final File SERIAL_FILENAME = TEMP_DIR.createFile("encogtest.ser");
		
	private SOM create()
	{
		SOM network = new SOM(4,2);
		return network;
	}
	
	public void testPersistEG()
	{
		SOM network = create();

		EncogDirectoryPersistence.saveObject(EG_FILENAME, network);
		SOM network2 = (SOM)EncogDirectoryPersistence.loadObject(EG_FILENAME);
		
		validate(network2);
	}
	
	public void testPersistSerial() throws IOException, ClassNotFoundException
	{
		SOM network = create();		
		SerializeObject.save(SERIAL_FILENAME, network);
		SOM network2 = (SOM)SerializeObject.load(SERIAL_FILENAME);
				
		validate(network2);
	}
	
	private void validate(SOM network)
	{
		Assert.assertEquals(4, network.getInputNeuronCount());
		Assert.assertEquals(2, network.getOutputNeuronCount());
		Assert.assertEquals(8, network.getWeights().toPackedArray().length);
	}
	
	@Override
	protected void tearDown() throws Exception {
		super.tearDown();
		TEMP_DIR.dispose();
	}
}
