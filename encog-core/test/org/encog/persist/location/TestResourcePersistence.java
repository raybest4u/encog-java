package org.encog.persist.location;

import java.io.InputStream;

import org.encog.util.file.Directory;
import org.junit.Assert;
import org.junit.Test;

public class TestResourcePersistence {
	
	@Test
	public void testResource() throws Exception
	{
		ResourcePersistence location = new ResourcePersistence("org/encog/data/testresource");
		InputStream is = location.createInputStream();
		String str = Directory.readStream(is);
		Assert.assertEquals("This is a test resource",str);
	}
}
