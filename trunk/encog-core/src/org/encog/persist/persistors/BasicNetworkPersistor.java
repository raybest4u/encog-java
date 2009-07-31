/*
 * Encog Artificial Intelligence Framework v2.x
 * Java Version
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

package org.encog.persist.persistors;

import java.util.HashMap;
import java.util.Map;

import org.encog.EncogError;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.logic.ART1Logic;
import org.encog.neural.networks.logic.BAMLogic;
import org.encog.neural.networks.logic.BoltzmannLogic;
import org.encog.neural.networks.logic.FeedforwardLogic;
import org.encog.neural.networks.logic.HopfieldLogic;
import org.encog.neural.networks.logic.NeuralLogic;
import org.encog.neural.networks.logic.SimpleRecurrentLogic;
import org.encog.neural.networks.synapse.Synapse;
import org.encog.parse.tags.read.ReadXML;
import org.encog.parse.tags.write.WriteXML;
import org.encog.persist.EncogPersistedCollection;
import org.encog.persist.EncogPersistedObject;
import org.encog.persist.Persistor;

/**
 * The Encog persistor used to persist the BasicNetwork class.
 * 
 * @author jheaton
 */
public class BasicNetworkPersistor implements Persistor {

	/**
	 * The layers tag.
	 */
	public static final String TAG_LAYERS = "layers";
	
	/**
	 * The synapses tag.
	 */
	public static final String TAG_SYNAPSES = "synapses";
	
	/**
	 * The synapse tag.
	 */
	public static final String TAG_SYNAPSE = "synapse";
	
	/**
	 * The properties tag.
	 */
	public static final String TAG_PROPERTIES = "properties";
	
	public static final String TAG_LOGIC = "logic";
	
	/**
	 * The layer synapse.
	 */
	public static final String TAG_LAYER = "layer";
	
	public static final String TAG_PROPERTY = "Property";
	
	/**
	 * The id attribute.
	 */
	public static final String ATTRIBUTE_ID = "id";
	
	public static final String ATTRIBUTE_NAME = "name";
	public static final String ATTRIBUTE_VALUE = "value";
	
	/**
	 * The type attribute.
	 */
	public static final String ATTRIBUTE_TYPE = "type";
	
	/**
	 * The input layer type.
	 */
	public static final String ATTRIBUTE_TYPE_INPUT = "input";
	
	/**
	 * The output layer type.
	 */
	public static final String ATTRIBUTE_TYPE_OUTPUT = "output";
	
	/**
	 * The hidden layer type.
	 */
	public static final String ATTRIBUTE_TYPE_HIDDEN = "hidden";
	
	/**
	 * The both layer type.
	 */
	public static final String ATTRIBUTE_TYPE_BOTH = "both";
	
	/**
	 * The unknown layer type.
	 */
	public static final String ATTRIBUTE_TYPE_UNKNOWN = "unknown";
	
	/**
	 * The from attribute.
	 */
	public static final String ATTRIBUTE_FROM = "from";
	
	/**
	 * The to attribute.
	 */
	public static final String ATTRIBUTE_TO = "to";

	/**
	 * The network that is being loaded.
	 */
	private BasicNetwork currentNetwork;
	
	/**
	 * A mapping from layers to index numbers.
	 */
	private final Map<Layer, Integer> layer2index 
		= new HashMap<Layer, Integer>();
	
	/**
	 * A mapping from index numbers to layers.
	 */
	private final Map<Integer, Layer> index2layer 
		= new HashMap<Integer, Layer>();

	/**
	 * Handle any layers that should be loaded.
	 * @param in The XML reader.
	 */
	private void handleLayers(final ReadXML in) {
		final String end = in.getTag().getName();
		while (in.readToTag()) {
			if (in.is(BasicNetworkPersistor.TAG_LAYER, true)) {
				final int num = in.getTag().getAttributeInt(
						BasicNetworkPersistor.ATTRIBUTE_ID);
				final String type = in.getTag().getAttributeValue(
						BasicNetworkPersistor.ATTRIBUTE_TYPE);
				in.readToTag();
				final Persistor persistor = PersistorUtil.createPersistor(in
						.getTag().getName());
				final Layer layer = (Layer) persistor.load(in);
				this.index2layer.put(num, layer);
				if (type.equals(BasicNetworkPersistor.ATTRIBUTE_TYPE_INPUT)) {
					this.currentNetwork.setInputLayer(layer);
				} else if (type
						.equals(BasicNetworkPersistor.ATTRIBUTE_TYPE_OUTPUT)) {
					this.currentNetwork.setOutputLayer(layer);
				} else if (type
						.equals(BasicNetworkPersistor.ATTRIBUTE_TYPE_BOTH)) {
					this.currentNetwork.setInputLayer(layer);
					this.currentNetwork.setOutputLayer(layer);
				}
			}
			if (in.is(end, false)) {
				break;
			}
		}
	}

	/**
	 * Process any synapses that should be loaded.
	 * @param in The XML reader.
	 */
	private void handleSynapses(final ReadXML in) {
		final String end = in.getTag().getName();
		while (in.readToTag()) {
			if (in.is(BasicNetworkPersistor.TAG_SYNAPSE, true)) {
				final int from = in.getTag().getAttributeInt(
						BasicNetworkPersistor.ATTRIBUTE_FROM);
				final int to = in.getTag().getAttributeInt(
						BasicNetworkPersistor.ATTRIBUTE_TO);
				in.readToTag();
				final Persistor persistor = PersistorUtil.createPersistor(in
						.getTag().getName());
				final Synapse synapse = (Synapse) persistor.load(in);
				synapse.setFromLayer(this.index2layer.get(from));
				synapse.setToLayer(this.index2layer.get(to));
				synapse.getFromLayer().addSynapse(synapse);
			}
			if (in.is(end, false)) {
				break;
			}
		}
	}

	/**
	 * Load the specified Encog object from an XML reader.
	 * 
	 * @param in
	 *            The XML reader to use.
	 * @return The loaded object.
	 */
	public EncogPersistedObject load(final ReadXML in) {

		final String name = in.getTag().getAttributes().get(
				EncogPersistedCollection.ATTRIBUTE_NAME);
		final String description = in.getTag().getAttributes().get(
				EncogPersistedCollection.ATTRIBUTE_DESCRIPTION);

		this.currentNetwork = new BasicNetwork();
		this.currentNetwork.setName(name);
		this.currentNetwork.setDescription(description);

		while (in.readToTag()) {
			if (in.is(BasicNetworkPersistor.TAG_LAYERS, true)) {
				handleLayers(in);
			} else if (in.is(BasicNetworkPersistor.TAG_SYNAPSES, true)) {
				handleSynapses(in);
			} else if( in.is(BasicNetworkPersistor.TAG_PROPERTIES, true)) {
				handleProperties(in);
			}
			else if( in.is(BasicNetworkPersistor.TAG_LOGIC, true)) {
				handleLogic(in);
			}

		}
		this.currentNetwork.getStructure().finalizeStructure();
		return this.currentNetwork;
	}

	private void handleLogic(ReadXML in) {
		String value = in.readTextToTag();
		if( value.equalsIgnoreCase("ART1Logic") )
		{
			this.currentNetwork.setLogic(new ART1Logic());
		} else if( value.equalsIgnoreCase("BAMLogic") )
		{
			this.currentNetwork.setLogic(new BAMLogic());
		}
		else if( value.equalsIgnoreCase("BoltzmannLogic") )
		{
			this.currentNetwork.setLogic(new BoltzmannLogic());
		} else if( value.equalsIgnoreCase("FeedforwardLogic") )
		{
			this.currentNetwork.setLogic(new FeedforwardLogic());
		} else if( value.equalsIgnoreCase("HopfieldLogic") )
		{
			this.currentNetwork.setLogic(new HopfieldLogic());
		} else if( value.equalsIgnoreCase("SimpleRecurrentLogic") )
		{
			this.currentNetwork.setLogic(new SimpleRecurrentLogic());
		} else
		{
			try {
				NeuralLogic logic = (NeuralLogic) Class.forName(value).newInstance();
				this.currentNetwork.setLogic(logic);
			} catch (ClassNotFoundException e) {
				throw new EncogError(e);
			} catch (InstantiationException e) {
				throw new EncogError(e);
			} catch (IllegalAccessException e) {
				throw new EncogError(e);
			}
		}
	}

	private void handleProperties(ReadXML in) {
		final String end = in.getTag().getName();
		while (in.readToTag()) {
			if (in.is(BasicNetworkPersistor.TAG_PROPERTY, true)) {
				final String name = in.getTag().getAttributeValue(
						BasicNetworkPersistor.ATTRIBUTE_NAME);
				final String type = in.getTag().getAttributeValue(
						BasicNetworkPersistor.ATTRIBUTE_TYPE);
				
				String value = in.readTextToTag();
				
				if(type.equals("D"))
				{
					this.currentNetwork.setProperty(name, Double.parseDouble(value));
				} else if(type.equals("L"))
				{
					this.currentNetwork.setProperty(name, Long.parseLong(value));
				} else if(type.equals("S"))
				{
					this.currentNetwork.setProperty(name, value);
				}
			}
			if (in.is(end, false)) {
				break;
			}
		}
		
	}

	/**
	 * Save the specified Encog object to an XML writer.
	 * 
	 * @param obj
	 *            The object to save.
	 * @param out
	 *            The XML writer to save to.
	 */
	public void save(final EncogPersistedObject obj, final WriteXML out) {
		PersistorUtil.beginEncogObject(EncogPersistedCollection.TYPE_BASIC_NET,
				out, obj, true);
		this.currentNetwork = (BasicNetwork) obj;

		this.currentNetwork.getStructure().finalizeStructure();

		// save the layers
		out.beginTag(BasicNetworkPersistor.TAG_LAYERS);
		saveLayers(out);
		out.endTag();

		// save the structure of these layers
		out.beginTag(BasicNetworkPersistor.TAG_SYNAPSES);
		saveSynapses(out);
		out.endTag();
		
		saveProperties(out);
		saveLogic(out);
		
		out.endTag();
	}
	
	private void saveLogic(WriteXML out)
	{
		out.beginTag(BasicNetworkPersistor.TAG_LOGIC);
		NeuralLogic logic = this.currentNetwork.getLogic();
		if( logic instanceof FeedforwardLogic ||
			logic instanceof SimpleRecurrentLogic ||
			logic instanceof BoltzmannLogic ||
			logic instanceof ART1Logic ||
			logic instanceof BAMLogic ||
			logic instanceof HopfieldLogic )
		{
			out.addText(logic.getClass().getSimpleName());
		}
		else
			out.addText(logic.getClass().getName());
		out.endTag();
	}
	
	private void saveProperties(WriteXML out)
	{
		// save any properties
		out.beginTag(BasicNetworkPersistor.TAG_PROPERTIES);
		for( String key: this.currentNetwork.getProperties().keySet() )
		{
			Object value = this.currentNetwork.getProperties().get(key);
			String type;
			if( value instanceof Double )
				type = "D";
			else if( value instanceof Long )
				type = "L";
			else 
				type = "S";
			
			
			out.addAttribute(BasicNetworkPersistor.ATTRIBUTE_NAME,key);
			out.addAttribute(BasicNetworkPersistor.ATTRIBUTE_TYPE,type);
			out.beginTag(BasicNetworkPersistor.TAG_PROPERTY);
			out.addText(value.toString());
			out.endTag();
		}
		out.endTag();
	}

	/**
	 * Save the layers to the specified XML writer.
	 * @param out The XML writer.
	 */
	private void saveLayers(final WriteXML out) {
		int current = 1;
		for (final Layer layer 
				: this.currentNetwork.getStructure().getLayers()) {
			String type;

			if (this.currentNetwork.isInput(layer)
					&& this.currentNetwork.isOutput(layer)) {
				type = BasicNetworkPersistor.ATTRIBUTE_TYPE_BOTH;
			} else if (this.currentNetwork.isInput(layer)) {
				type = BasicNetworkPersistor.ATTRIBUTE_TYPE_INPUT;
			} else if (this.currentNetwork.isOutput(layer)) {
				type = BasicNetworkPersistor.ATTRIBUTE_TYPE_OUTPUT;
			} else if (this.currentNetwork.isHidden(layer)) {
				type = BasicNetworkPersistor.ATTRIBUTE_TYPE_HIDDEN;
			} else {
				type = BasicNetworkPersistor.ATTRIBUTE_TYPE_UNKNOWN;
			}

			out.addAttribute(BasicNetworkPersistor.ATTRIBUTE_ID, "" + current);
			out.addAttribute(BasicNetworkPersistor.ATTRIBUTE_TYPE, type);
			out.beginTag(BasicNetworkPersistor.TAG_LAYER);
			final Persistor persistor = layer.createPersistor();
			persistor.save(layer, out);
			out.endTag();
			this.layer2index.put(layer, current);
			current++;
		}
	}

	/**
	 * Save the synapses to the specified XML writer.
	 * @param out The XML writer.
	 */
	private void saveSynapses(final WriteXML out) {
		for (final Synapse synapse : this.currentNetwork.getStructure()
				.getSynapses()) {
			out.addAttribute(BasicNetworkPersistor.ATTRIBUTE_FROM, ""
					+ this.layer2index.get(synapse.getFromLayer()));
			out.addAttribute(BasicNetworkPersistor.ATTRIBUTE_TO, ""
					+ this.layer2index.get(synapse.getToLayer()));
			out.beginTag(BasicNetworkPersistor.TAG_SYNAPSE);
			final Persistor persistor = synapse.createPersistor();
			persistor.save(synapse, out);
			out.endTag();
		}
	}
}
