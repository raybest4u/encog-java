package org.encog.app.quant;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.encog.app.quant.filter.FilterCSV;
import org.encog.util.csv.CSVFormat;

import junit.framework.Assert;
import junit.framework.TestCase;

public class TestFilter extends TestCase {

    public final static String INPUT_NAME = "test.csv";
    public final static String OUTPUT_NAME = "test2.csv";

    public void generateTestFileHeadings(boolean header) throws IOException
    {
    	PrintWriter tw = new PrintWriter(new FileWriter(INPUT_NAME));

        if (header)
        {
            tw.println("a,b");
        }
        tw.println("one,1");
        tw.println("two,1");
        tw.println("three,1");
        tw.println("four,2");
        tw.println("five,1");
        tw.println("six,1");

        // close the stream
        tw.close();
    }

    public void testFilterCSVHeaders() throws IOException
    {
        generateTestFileHeadings(true);
        FilterCSV norm = new FilterCSV();
        norm.analyze(INPUT_NAME, true, CSVFormat.ENGLISH);
        norm.exclude(1, "1");
        norm.process(OUTPUT_NAME);

        BufferedReader tr = new BufferedReader(new FileReader(OUTPUT_NAME));
        Assert.assertEquals("\"a\",\"b\"", tr.readLine());
        Assert.assertEquals("four,2", tr.readLine());
        tr.close();

        (new File(INPUT_NAME)).delete();
        (new File(OUTPUT_NAME)).delete();
    }

    public void TestFilterCSVNoHeaders() throws IOException
    {
        generateTestFileHeadings(false);
        FilterCSV norm = new FilterCSV();
        norm.analyze(INPUT_NAME, false, CSVFormat.ENGLISH);
        norm.exclude(1, "1");
        norm.process(OUTPUT_NAME);

        BufferedReader tr = new BufferedReader(new FileReader(OUTPUT_NAME));
        Assert.assertEquals("four,2", tr.readLine());
        tr.close();

        (new File(INPUT_NAME)).delete();
        (new File(OUTPUT_NAME)).delete();
    }
	
}
