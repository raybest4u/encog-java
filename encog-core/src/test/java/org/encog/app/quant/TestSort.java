package org.encog.app.quant;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import junit.framework.TestCase;

import org.encog.app.csv.sort.SortCSV;
import org.encog.app.csv.sort.SortType;
import org.encog.app.csv.sort.SortedField;
import org.encog.util.csv.CSVFormat;
import org.junit.Assert;

public class TestSort extends TestCase {

	public static final File INPUT_NAME = new File("test.csv");
    public static final File OUTPUT_NAME = new File("test2.csv");


    public void generateTestFileHeadings(boolean header) throws IOException
    {
    	PrintWriter tw = new PrintWriter(new FileWriter(INPUT_NAME));

        if (header)
        {
            tw.println("a,b");
        }

        tw.println("five,5");
        tw.println("four,4");
        tw.println("two,2");
        tw.println("three,3");                      
        tw.println("six,6");
        tw.println("one,1");

        // close the stream
        tw.close();
    }

    public void testSortHeaders() throws IOException
    {
        generateTestFileHeadings(true);
        SortCSV norm = new SortCSV();
        norm.getSortOrder().add(new SortedField(1,SortType.SortString,true));
        norm.process(INPUT_NAME,OUTPUT_NAME,true,CSVFormat.ENGLISH);

        BufferedReader tr = new BufferedReader(new FileReader(OUTPUT_NAME));

        Assert.assertEquals("\"a\",\"b\"", tr.readLine());
        Assert.assertEquals("\"one\",1", tr.readLine());
        Assert.assertEquals("\"two\",2", tr.readLine());
        Assert.assertEquals("\"three\",3", tr.readLine());
        Assert.assertEquals("\"four\",4", tr.readLine());
        Assert.assertEquals("\"five\",5", tr.readLine());
        Assert.assertEquals("\"six\",6", tr.readLine());
        Assert.assertNull(tr.readLine());


        tr.close();

        (new File("test.csv")).delete();
        (new File("test2.csv")).delete();
    }
    
    public void testSortNoHeaders() throws IOException
    {
        generateTestFileHeadings(false);
        SortCSV norm = new SortCSV();
        norm.getSortOrder().add(new SortedField(1,SortType.SortInteger,true));
        norm.setProduceOutputHeaders(false);
        norm.process(INPUT_NAME,OUTPUT_NAME,false,CSVFormat.ENGLISH);

        BufferedReader tr = new BufferedReader(new FileReader(OUTPUT_NAME));

        Assert.assertEquals("\"one\",1", tr.readLine());
        Assert.assertEquals("\"two\",2", tr.readLine());
        Assert.assertEquals("\"three\",3", tr.readLine());
        Assert.assertEquals("\"four\",4", tr.readLine());
        Assert.assertEquals("\"five\",5", tr.readLine());
        Assert.assertEquals("\"six\",6", tr.readLine());
        Assert.assertNull(tr.readLine());


        tr.close();

        (new File("test.csv")).delete();
        (new File("test2.csv")).delete();
    }


	
	
}
