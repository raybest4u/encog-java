package org.encog.script.basic;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.encog.util.BasicEvaluate;

public class TestBasicLoop extends TestCase {
	
	public void testWhile()
	{
		String str = BasicEvaluate.evaluate("test-while");
		Assert.assertEquals("1.02.03.04.05.06.07.08.09.0", str);	
	}
	
	public void testDo1()
	{
		String str = BasicEvaluate.evaluate("test-do1");
		Assert.assertEquals("1.02.03.04.05.06.07.08.09.0", str);	
	}
	
	public void testDo2()
	{
		String str = BasicEvaluate.evaluate("test-do2");
		Assert.assertEquals("1.02.03.04.05.06.07.08.09.0", str);	
	}
	
	public void testFor()
	{
		String str = BasicEvaluate.evaluate("test-for");
		Assert.assertEquals("1.02.03.04.05.06.07.08.09.010.01.03.05.07.09.010.09.08.07.06.05.04.03.02.01.0", str);	
	}
}
