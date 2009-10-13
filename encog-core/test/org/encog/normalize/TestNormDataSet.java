package org.encog.normalize;

import junit.framework.TestCase;

import org.encog.NullStatusReportable;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.normalize.input.InputField;
import org.encog.normalize.input.InputFieldArray2D;
import org.encog.normalize.input.InputFieldNeuralDataSet;
import org.encog.normalize.output.OutputFieldRangeMapped;
import org.encog.normalize.target.NormalizationStorageArray2D;
import org.junit.Assert;

public class TestNormDataSet extends TestCase {
	
	public static final double[][] ARRAY_2D = { {1.0,2.0,3.0,4.0,5.0},
		{6.0,7.0,8.0,9.0} };
	
	
	public void testDataSet()
	{
		InputField a,b;
		double[][] arrayOutput = new double[2][2];
		
		BasicNeuralDataSet dataset = new BasicNeuralDataSet(ARRAY_2D,null);
		
		NormalizationStorageArray2D target = new NormalizationStorageArray2D(arrayOutput);
		
		Normalization norm = new Normalization();
		norm.setReport(new NullStatusReportable());
		norm.setTarget(target);
		norm.addInputField(a = new InputFieldNeuralDataSet(dataset,0));
		norm.addInputField(b = new InputFieldNeuralDataSet(dataset,1));
		norm.addOutputField(new OutputFieldRangeMapped(a,0.1,0.9));
		norm.addOutputField(new OutputFieldRangeMapped(b,0.1,0.9));
		norm.process();
		Assert.assertEquals(arrayOutput[0][0],0.1,0.1);
		Assert.assertEquals(arrayOutput[1][0],0.9,0.1);
		Assert.assertEquals(arrayOutput[0][1],0.1,0.1);
		Assert.assertEquals(arrayOutput[1][1],0.9,0.1);
		
	}
}
