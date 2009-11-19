package org.encog.examples.neural.forest;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.buffer.BufferedNeuralDataSet;
import org.encog.neural.data.csv.CSVNeuralDataSet;
import org.encog.neural.data.market.MarketNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.normalize.DataNormalization;
import org.encog.normalize.output.nominal.OutputEquilateral;
import org.encog.persist.EncogPersistedCollection;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.math.Equilateral;

public class Evaluate {
	
	private int[] treeCount = new int[10];
	private int[] treeCorrect = new int[10];
	
	public void keepScore(int actual,int ideal)
	{
		treeCount[ideal]++;
		if(actual==ideal)
			treeCorrect[ideal]++;	
	}
	
	public BasicNetwork loadNetwork()
	{
		File file = Constant.TRAINED_NETWORK_FILE;
		
		if( !file.exists() )
		{
			System.out.println("Can't read file: " + file.getAbsolutePath() );
			return null;
		}
		
		EncogPersistedCollection encog = new EncogPersistedCollection(file);					
		BasicNetwork network = (BasicNetwork) encog.find(Constant.TRAINED_NETWORK_NAME);
	
		if( network==null )
		{
			System.out.println("Can't find network resource: " + Constant.TRAINED_NETWORK_NAME );
			return null;
		}
				
		return network;
	}
	

	
	public DataNormalization loadNormalization()
	{
		File file = Constant.TRAINED_NETWORK_FILE;
		
		EncogPersistedCollection encog = new EncogPersistedCollection(file);
		
		DataNormalization norm = (DataNormalization) encog.find(Constant.NORMALIZATION_NAME);
		if( norm==null )
		{
			System.out.println("Can't find normalization resource: " + Constant.NORMALIZATION_NAME );
			return null;
		}
				
		return norm;
	}
	
	public int determineTreeType(OutputEquilateral eqField, NeuralData output)
	{
		int result = 0;
		
		if( eqField!=null )
		{
			result = eqField.getEquilateral().decode(output.getData());			
		}
		else
		{
			double maxOutput = Double.NEGATIVE_INFINITY;
			result = -1;
			
			for(int i=0;i<output.size();i++)
			{
				if( output.getData(i)>maxOutput )
				{
					maxOutput = output.getData(i);
					result = i;
				}
			}
		}
			
		return result;
	}
	
	public void evaluate2()
	{
		BasicNetwork network = loadNetwork();
		BufferedNeuralDataSet trainingSet = new BufferedNeuralDataSet(Constant.BINARY_FILE);
		System.out.println("Error:" + network.calculateError(trainingSet));
	}
	
	public void evaluate1()
	{
		int[] count = new int[7];
		
		Equilateral equ = new Equilateral(7,0.9,0.1);
		BasicNetwork network = loadNetwork();
		BufferedNeuralDataSet trainingSet = new BufferedNeuralDataSet(Constant.BINARY_FILE);
		for(NeuralDataPair pair : trainingSet)
		{
			NeuralData actual = network.compute(pair.getInput());
			int tree = equ.decode(actual.getData());
			count[tree]++;
		}
		
		for(int i=0;i<count.length;i++)
		{
			System.out.println(i+"-"+count[i]);
		}
	}
	
	public void evaluate3()
	{
		int[] count = new int[7];
		int correct = 0;
		int total = 0;
				
		Equilateral equ = new Equilateral(7,0.9,0.1);
		BasicNetwork network = loadNetwork();
		DataNormalization norm = loadNormalization();
		
		NeuralDataSet trainingSet = new CSVNeuralDataSet(
				Constant.BALANCE_FILE.toString(),54,0,false);
		
		for(NeuralDataPair pair : trainingSet)
		{
			total++;
			NeuralData input = norm.buildForNetworkInput(pair.getInput().getData());
			NeuralData actual = network.compute(input);
			int actualTree = equ.decode(actual.getData());
			int idealTree = (int)(pair.getInput().getData(53))-1;
			count[actualTree]++;
		}
		
		for(int i=0;i<count.length;i++)
		{
			System.out.println(i+"-"+count[i]);
		}
	}
	
	public void evaluate()
	{
		BasicNetwork network = loadNetwork();
		DataNormalization norm = loadNormalization();
		
		ReadCSV csv = new ReadCSV(Constant.BALANCE_FILE.toString(),false,',');
		double[] input = new double[norm.getInputFields().size()];
		OutputEquilateral eqField = (OutputEquilateral)norm.findOutputField(OutputEquilateral.class, 0);
		
		int correct = 0;
		int total = 0;
		while(csv.next())
		{
			total++;
			for(int i=0;i<input.length;i++)
			{
				input[i] = csv.getDouble(i);
			}
			NeuralData inputData = norm.buildForNetworkInput(input);
			NeuralData output = network.compute(inputData);
			int coverTypeActual = determineTreeType(eqField,output);
			int coverTypeIdeal = (int)csv.getDouble(54)-1;
			
			keepScore(coverTypeActual,coverTypeIdeal);
			
			if( coverTypeActual==coverTypeIdeal ) {
				correct++;
			}
			//System.out.println(coverTypeActual + " - " + coverTypeIdeal );
		}
		
		System.out.println("Total cases:" + total);
		System.out.println("Correct cases:" + correct);
		double percent = (double)correct/(double)total;
		System.out.println("Correct percent:" + (percent*100.0));
		for(int i=0;i<10;i++)
		{
			double p = ((double)this.treeCorrect[i] / (double)this.treeCount[i])*100.0;
			System.out.println("Tree #" + i + " - Correct" + this.treeCorrect[i] + "/" + treeCount[i] + "(" + p + ")" );
		}
	}
}
