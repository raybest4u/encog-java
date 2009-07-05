/*
 * Encog Workbench v2.x
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2009, Heaton Research Inc., and individual contributors.
 * See the copyright.txt in the distribution for a full listing of 
 * individual contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */

package org.encog.workbench.process.generate;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.layers.ContextLayer;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.layers.RadialBasisFunctionLayer;
import org.encog.neural.networks.synapse.Synapse;

public class GenerateJava extends BasicGenerate {

	
	public String generate(final BasicNetwork network) {

		this.setNetwork(network);

		addClass(this.getNetwork().getClass());

		this.addNewLine();
		this.addLine("public class EncogGeneratedClass {");
		this.addNewLine();

		generateMain();
		this.addLine("}\n");

		final String importStr = generateImports();

		return importStr + this.getSource().toString();
	}



	private String generateImports() {
		final StringBuilder results = new StringBuilder();
		for (final String c : this.getImports()) {
			results.append("import ");
			results.append(c);
			results.append(";\n");
		}
		return results.toString();
	}
	
	public void generateLayer(Layer previousLayer, Layer currentLayer)
	{		
		addClass(Layer.class);
		this.appendToLine("Layer ");
		this.appendToLine(nameLayer(currentLayer));
		this.appendToLine(" = ");
		
		this.addObject(currentLayer);
		
		if( currentLayer instanceof ContextLayer )
		{
			this.addObject(currentLayer.getActivationFunction());
			this.appendToLine("new ContextLayer( new ");
			this.appendToLine(currentLayer.getActivationFunction().getClass().getSimpleName());
			this.appendToLine("()");
			this.appendToLine(",");
			this.appendToLine(currentLayer.hasThreshold()?"true":"false");
			this.appendToLine(",");
			this.appendToLine(""+currentLayer.getNeuronCount());
			this.appendToLine(");");
		}
		else if( currentLayer instanceof RadialBasisFunctionLayer )
		{
			this.appendToLine("new RadialBasisFunctionLayer(");
			this.appendToLine(""+currentLayer.getNeuronCount());
			this.appendToLine(");");
		}
		else if( currentLayer instanceof BasicLayer )
		{
			this.addObject(currentLayer.getActivationFunction());
			this.appendToLine("new BasicLayer( new ");
			this.appendToLine(currentLayer.getActivationFunction().getClass().getSimpleName());
			this.appendToLine("()");
			this.appendToLine(",");
			this.appendToLine(currentLayer.hasThreshold()?"true":"false");
			this.appendToLine(",");
			this.appendToLine(""+currentLayer.getNeuronCount());
			this.appendToLine(");");
		}
		
		this.addLine();
		
		if( previousLayer==null) {
			this.appendToLine("\nnetwork.addLayer(");
			this.appendToLine(nameLayer(currentLayer));
			this.appendToLine(");");
		}
		else
		{
			this.addNewLine();
			this.appendToLine(nameLayer(previousLayer));
			this.appendToLine(".addNext(");
			this.appendToLine(nameLayer(currentLayer));
			this.appendToLine(");");
		}
		
		this.addLine();
		
		// next layers
		for(Synapse nextSynapse: currentLayer.getNext() )
		{
			Layer nextLayer = nextSynapse.getToLayer();
			if( this.getLayerMap().containsKey(nextLayer))
			{
				this.appendToLine(nameLayer(currentLayer));
				this.appendToLine(".addNext(");
				this.appendToLine(nameLayer(nextLayer));
				this.appendToLine(");\n");
				this.addLine();
			}
			else
			{
				generateLayer( currentLayer, nextSynapse.getToLayer());
			}
		}

	}

	public void generateMain() {
		this.addLine("  public static void main(final String args[]) {");
		this.addNewLine();

		this.addLine("  BasicNetwork network = new BasicNetwork();");

		generateLayer(null, this.getNetwork().getInputLayer());
		
		this.addLine("  network.getStructure().finalizeStructure();");
		this.addLine("  network.reset();");
		
		this.addLine("  }");

	}
}
