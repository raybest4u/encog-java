package org.encog.workbench.process;

import org.encog.neural.activation.ActivationTANH;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.HopfieldPattern;
import org.encog.neural.pattern.SOMPattern;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.dialogs.createnetwork.CreateElmanDialog;
import org.encog.workbench.dialogs.createnetwork.CreateFeedforward;
import org.encog.workbench.dialogs.createnetwork.CreateHopfieldDialog;
import org.encog.workbench.dialogs.createnetwork.CreateNeuralNetworkDialog;
import org.encog.workbench.dialogs.createnetwork.CreateSOMDialog;
import org.encog.workbench.dialogs.createnetwork.NeuralNetworkType;

public class CreateNeuralNetwork {
	
	public static void process(String name)
	{
		BasicNetwork network = null;
		CreateNeuralNetworkDialog dialog = new CreateNeuralNetworkDialog(EncogWorkBench.getInstance().getMainWindow());
		dialog.setType(NeuralNetworkType.Empty);
		if( dialog.process() )
		{
			switch(dialog.getType())
			{
				case Empty:
					network = createEmpty(name);
					break;
				case Feedforward:
					network = createFeedForward(name);
					break;
				case SOM:
					network = createSOM(name);
					break;
				case Hopfield:
					network = createHopfield(name);
					break;
				case Elman:
					network = createElman(name);
					break;
				case Jordan:
					network = createJordan(name);
					break;
				case RBF:
					network = createRBF(name);
					break;
				default:
					network = createEmpty(name);
					break;
			}
			
			if( network!=null )
			{
			EncogWorkBench.getInstance().getCurrentFile().add(name,network);
			EncogWorkBench.getInstance().getMainWindow().redraw();
			}
		}
	}

	private static BasicNetwork createRBF(String name) {
		return null;
		
	}

	private static BasicNetwork createJordan(String name) {
		return null;
		
	}

	private static BasicNetwork createElman(String name) {
		CreateElmanDialog dialog = new CreateElmanDialog(EncogWorkBench.getInstance().getMainWindow());
		if( dialog.process())
		{
		ElmanPattern elman = new ElmanPattern();
		elman.setInputNeurons(dialog.getInputCount().getValue());
		elman.addHiddenLayer(dialog.getHiddenCount().getValue());
		elman.setOutputNeurons(dialog.getOutputCount().getValue());
		return elman.generate();
		}
		else
			return null;
		
	}

	private static BasicNetwork createHopfield(String name) {
		CreateHopfieldDialog dialog = new CreateHopfieldDialog(EncogWorkBench.getInstance().getMainWindow());
		if( dialog.process())
		{
		HopfieldPattern hopfield = new HopfieldPattern();
		hopfield.setInputNeurons(dialog.getNeuronCount().getValue());
		return hopfield.generate();
		}
		else
			return null;
		
	}

	private static BasicNetwork createSOM(String name) {
		CreateSOMDialog dialog = new CreateSOMDialog(EncogWorkBench.getInstance().getMainWindow());
		if( dialog.process())
		{
		SOMPattern som = new SOMPattern();
		som.setInputNeurons(dialog.getInputCount().getValue());
		som.setOutputNeurons(dialog.getOutputCount().getValue());
		return som.generate();
		}
		else
			return null;
	}

	private static BasicNetwork createFeedForward(String name) {
		CreateFeedforward dialog = new CreateFeedforward(EncogWorkBench.getInstance().getMainWindow());
		if( dialog.process())
		{
			FeedForwardPattern feedforward = new FeedForwardPattern();
			feedforward.setInputNeurons(dialog.getInputCount().getValue());
			for(int i=0;i<dialog.getHidden().getModel().size();i++)
			{
				String str = (String)dialog.getHidden().getModel().getElementAt(i);
				int i1 = str.indexOf(':');
				int i2 = str.indexOf("neur");
				if( i1!=-1 && i2!=-1 )
				{
					str = str.substring(i1+1,i2).trim();
					int neuronCount = Integer.parseInt(str);
					feedforward.addHiddenLayer(neuronCount);
				}
			}
			feedforward.setInputNeurons(dialog.getInputCount().getValue());
			feedforward.setOutputNeurons(dialog.getOutputCount().getValue());
			feedforward.setActivationFunction(new ActivationTANH());
			return feedforward.generate();
		}
		return null;
		
	}

	private static BasicNetwork createEmpty(String name) {
		BasicNetwork network = new BasicNetwork();
		network.setName(name);
		return network;
	}
}
