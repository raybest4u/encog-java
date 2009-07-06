package org.encog.workbench.process.generate;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.ContextLayer;
import org.encog.neural.networks.layers.Layer;

public abstract class BasicGenerate implements Generate {
	
	private final StringBuilder source = new StringBuilder();
	private final StringBuilder line = new StringBuilder();
	private BasicNetwork network;
	private final Set<String> imports = new TreeSet<String>();
	private int hiddenLayerNumber = 1;
	private int contextLayerNumber = 1;
	private int indent = 0;
	private final Map<Layer,String> layerMap = new HashMap<Layer,String>();

	public Set<String> getImports() {
		return imports;
	}

	public StringBuilder getSource() {
		return source;
	}
	
	public void addLine(final String line)
	{
		// handle the indent
		for(int i=0;i<this.indent;i++)
			source.append(' ');
		// add the line
		source.append(line);
		addNewLine();
	}
	
	public void addLine()
	{
		addLine(line.toString());
		line.setLength(0);
	}
	
	public void addNewLine()
	{
		source.append("\n");
	}
	
	public void appendToLine(String str)
	{
		line.append(str);
	}
	
	public void addObject(final Object obj) {
		this.imports.add(obj.getClass().getName());
	}
	
	@SuppressWarnings("unchecked")
	public void addClass(final Class c) {
		this.imports.add(c.getName());
	}
	
	public String nameLayer(Layer layer)
	{
		if( this.layerMap.containsKey(layer) )
		{
			return this.layerMap.get(layer);
		}
		
		StringBuilder result = new StringBuilder();
		
		if( layer==this.network.getInputLayer() )
		{
			result.append("inputLayer");
		}
		else if( layer==this.network.getOutputLayer() )
		{
			result.append("outputLayer");
		}
		else
		{
			if( layer instanceof ContextLayer )
			{
				result.append("contextLayer");
				result.append(this.contextLayerNumber++);
			}
			else
			{
				result.append("hiddenLayer");
				result.append(hiddenLayerNumber++);
			}
		}
		
		this.layerMap.put(layer, result.toString());
		return result.toString();
		
	}

	public BasicNetwork getNetwork() {
		return network;
	}

	public void setNetwork(BasicNetwork network) {
		this.network = network;
	}

	public Map<Layer, String> getLayerMap() {
		return layerMap;
	}
	
	
	public void forwardIndent()
	{
		this.indent+=2;
	}
	
	public void backwardIndent()
	{
		this.indent-=2;
		if( this.indent<0 )
			this.indent = 0;
	}
	
}
