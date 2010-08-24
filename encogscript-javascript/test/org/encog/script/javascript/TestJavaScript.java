package org.encog.script.javascript;


import org.encog.script.EncogScript;
import org.encog.script.StandardConsole;

import junit.framework.TestCase;

public class TestJavaScript extends TestCase {

	public void testHelloWorld()
	{
		EncogScript script = new EncogScript();
		script.setSource("print(\'Hello World\')\n");
		script.run(new StandardConsole());
	}
}
