package org.encog.app.quant;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.encog.app.quant.balance.BalanceCSV;
import org.encog.util.csv.CSVFormat;

public class TestBalanceCSV extends TestCase {
	public static final String INPUT_NAME = "test.csv";
	public static final String OUTPUT_NAME = "test2.csv";

	public void generateTestFile(boolean header) throws IOException {
		PrintWriter out = new PrintWriter(new FileWriter(INPUT_NAME));

		if (header) {
			out.println("a,b");
		}
		out.println("one,1");
		out.println("two,1");
		out.println("three,1");
		out.println("four,2");
		out.println("five,2");
		out.println("six,3");

		// close the stream
		out.close();
	}

	public void testBalanceCSVHeaders() throws IOException {
		generateTestFile(true);
		BalanceCSV norm = new BalanceCSV();
		norm.Analyze(INPUT_NAME, true, CSVFormat.ENGLISH);
		norm.Process(OUTPUT_NAME, 1, 2);

		BufferedReader tr = new BufferedReader(new FileReader(OUTPUT_NAME));

		Assert.assertEquals("\"a\",\"b\"", tr.readLine());
		Assert.assertEquals("one,1", tr.readLine());
		Assert.assertEquals("two,1", tr.readLine());
		Assert.assertEquals("four,2", tr.readLine());
		Assert.assertEquals("five,2", tr.readLine());
		Assert.assertEquals("six,3", tr.readLine());
		Assert.assertEquals(2, norm.getCounts().get("1").intValue());
		Assert.assertEquals(2, norm.getCounts().get("2").intValue());
		Assert.assertEquals(1, norm.getCounts().get("3").intValue());
		tr.close();

		(new File(INPUT_NAME)).delete();
		(new File(OUTPUT_NAME)).delete();

	}

	public void TestBalanceCSVNoHeaders() throws IOException {
		generateTestFile(false);
		BalanceCSV norm = new BalanceCSV();
		norm.Analyze(INPUT_NAME, false, CSVFormat.ENGLISH);
		norm.Process(OUTPUT_NAME, 1, 2);

		BufferedReader tr = new BufferedReader(new FileReader(OUTPUT_NAME));

		Assert.assertEquals("one,1", tr.readLine());
		Assert.assertEquals("two,1", tr.readLine());
		Assert.assertEquals("four,2", tr.readLine());
		Assert.assertEquals("five,2", tr.readLine());
		Assert.assertEquals("six,3", tr.readLine());
		Assert.assertEquals(2, norm.getCounts().get("1").intValue());
		Assert.assertEquals(2, norm.getCounts().get("2").intValue());
		Assert.assertEquals(1, norm.getCounts().get("3").intValue());
		tr.close();

		(new File(INPUT_NAME)).delete();
		(new File(OUTPUT_NAME)).delete();
	}

}
