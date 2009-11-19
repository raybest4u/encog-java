package org.encog.examples.neural.forest;

import org.encog.normalize.DataNormalization;
import org.encog.persist.EncogPersistedCollection;
import org.encog.util.logging.Logging;
/**
 * Elevation (meters)                       Elevation in meters, field 0
 * Aspect (azimuth)                         Aspect in degrees azimuth, field 1
 * Slope (degrees)                          Slope in degrees, field 2
 * Horizontal_Distance_To_Hydrology(meters) Horz Dist to nearest surface water features, field 3
 * Vertical_Distance_To_Hydrology (meters)  Vert Dist to nearest surface water features, field 4
 * Horizontal_Distance_To_Roadways (meters) Horz Dist to nearest roadway, field 5
 * Hillshade_9am (0 to 255 index)                       Hillshade index at 9am, summer solstice, field 6
 * Hillshade_Noon (0 to 255 index)                      Hillshade index at noon, summer soltice, field 7
 * Hillshade_3pm (0 to 255 index)                       Hillshade index at 3pm, summer solstice, field 8
 * Horizontal_Distance_To_Fire_Point(meters)Horz Dist to nearest wildfire ignition points, field 9
 * Wilderness_Area (4 binary columns)       0 (absence) or 1 (presence) Wilderness area designation, field 10-13
 * Soil_Type (40 binary columns)            0 (absence) or 1 (presence) Soil Type designation, field 14-54
 * Cover_Type (7 types) (integer)           1 to 7 Forest Cover Type designation, field 55-62
 *
 * Wilderness Areas:    
 *   1 -- Rawah Wilderness Area
 *   2 -- Neota Wilderness Area
 *   3 -- Comanche Peak Wilderness Area
 *   4 -- Cache la Poudre Wilderness Area
 *
 * Soil Types:          
 *   1 to 40 : based on the USFS Ecological Landtype Units for this study area.
 *
 * Forest Cover Types:  
 *   1 -- Spruce/Fir
 *   2 -- Lodgepole Pine
 *   3 -- Ponderosa Pine
 *   4 -- Cottonwood/Willow
 *   5 -- Aspen
 *   6 -- Douglas-fir
 *   7 -- Krummholz
 *
 */

public class ForestCover {
	
	public static void generate(boolean useOneOf)
	{
		GenerateData generate = new GenerateData();
		generate.step1();
		generate.step2();
		DataNormalization norm = generate.step3(useOneOf);
		
		EncogPersistedCollection encog = new EncogPersistedCollection(Constant.TRAINED_NETWORK_FILE);
		encog.add(Constant.NORMALIZATION_NAME, norm);
		
		//DataNormalization trainingNorm = generate.generateTraining(Constant.TRAINING_FILE,0, 0,2,4,useOneOf);
		//DataNormalization evaluateNorm = generate.generateIdeal(Constant.EVAL_FILE,0, 2,3,4);
		//EncogPersistedCollection encog = new EncogPersistedCollection(Constant.TRAINED_NETWORK_FILE);
		//encog.add(Constant.NORMALIZATION_NAME, trainingNorm);
		
		//System.out.println("Training samples:" + trainingNorm.getRecordCount());
		//System.out.println("Evaluate samples:" + evaluateNorm.getRecordCount());
	}
	
	public static void train(boolean useGUI)
	{		
		TrainNetwork program = new TrainNetwork();
		program.train(useGUI);
	}
	
	public static void evaluate()
	{
		Evaluate evaluate = new Evaluate();
		evaluate.evaluate();
	}
	
	public static void main(String args[])
	{
		if( args.length<2 )
		{
			System.out.println("Usage: ForestCover [generate/train/evaluate] [e/o]");
		}
		else
		{
			boolean useOneOf;
			
			if( args[1].toLowerCase().equals("e") )
				useOneOf = false;
			else
				useOneOf = true;
				
			Logging.stopConsoleLogging();
			if( args[0].equalsIgnoreCase("generate") )
				generate(useOneOf);
			else if( args[0].equalsIgnoreCase("train") )
				train(false);
			else if( args[0].equalsIgnoreCase("traingui") )
				train(true);
			else if( args[0].equalsIgnoreCase("evaluate") )
				evaluate();
		}
	}
}
