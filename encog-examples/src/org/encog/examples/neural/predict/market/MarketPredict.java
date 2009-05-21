package org.encog.examples.neural.predict.market;

import java.text.DecimalFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;

import javax.swing.text.NumberFormatter;

import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.market.MarketDataDescription;
import org.encog.neural.data.market.MarketDataType;
import org.encog.neural.data.market.MarketNeuralDataSet;
import org.encog.neural.data.market.loader.MarketLoader;
import org.encog.neural.data.market.loader.YahooFinanceLoader;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogPersistedCollection;
import org.encog.util.logging.Logging;


public class MarketPredict {
		
	
	enum Direction {
		up,
		down		
	};
	
	public static Direction determineDirection(double d)
	{
		/*if( Math.abs(d)<ZERO_TOLERANCE )
		{
			return Direction.flat;
		}*/
		if( d<0 )
			return Direction.down;
		else
			return Direction.up;
	}
	
	
	public static MarketNeuralDataSet grabData()
	{
		MarketLoader loader = new YahooFinanceLoader();
		MarketNeuralDataSet result = new MarketNeuralDataSet(
				loader, 
				Config.INPUT_WINDOW,
				Config.PREDICT_WINDOW);
		MarketDataDescription desc = new MarketDataDescription(
				Config.TICKER, 
				MarketDataType.ADJUSTED_CLOSE, 
				true,
				true);
		result.addDescription(desc);
		
		Calendar end = new GregorianCalendar();// end today		
		Calendar begin = (Calendar)end.clone();// begin 30 days ago
		begin.add(Calendar.DATE, -60);
		
		result.load(begin.getTime(), end.getTime());
		result.generate();
		
		return result;

	}
	
	public static void main(String args[])
	{
		Logging.stopConsoleLogging();
		
		EncogPersistedCollection encog = new EncogPersistedCollection(Config.FILENAME);					
		BasicNetwork network = (BasicNetwork) encog.find(Config.MARKET_NETWORK);
				
		MarketNeuralDataSet data = grabData();
		
		DecimalFormat format = new DecimalFormat("#0.00");
		
		int count = 0;
		int correct = 0;
		for(NeuralDataPair pair: data)
		{
			NeuralData input = pair.getInput();
			NeuralData actualData = pair.getIdeal();			
			NeuralData predictData = network.compute(input);
			
			double actual = actualData.getData(0);
			double predict = predictData.getData(0);			
			double diff = Math.abs(predict-actual);
			
			Direction actualDirection = determineDirection(actual);
			Direction predictDirection = determineDirection(predict);
			
			if( actualDirection==predictDirection )
				correct++;
			
			count++;
						
			System.out.println("Day " + count+":actual="
					+format.format(actual)+"("+actualDirection+")"
					+",predict=" 
					+format.format(predict)+"("+actualDirection+")"
					+",diff="+diff);			
			
		}
		double percent = (double)correct/(double)count;
		System.out.println("Direction correct:" + correct + "/" + count);
		System.out.println("Directional Accuracy:"+format.format(percent*100)+"%");
		
	}
	
}
