/*
 * Encog(tm) Core v2.5 
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2010 by Heaton Research Inc.
 * 
 * Released under the LGPL.
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
 * 
 * Encog and Heaton Research are Trademarks of Heaton Research, Inc.
 * For information on Heaton Research trademarks, visit:
 * 
 * http://www.heatonresearch.com/copyright.html
 */

package org.encog.persist.persistors;

import org.encog.neural.activation.ActivationFunction;
import org.encog.neural.networks.layers.ContextLayer;
import org.encog.parse.tags.read.ReadXML;
import org.encog.parse.tags.write.WriteXML;
import org.encog.persist.EncogPersistedCollection;
import org.encog.persist.EncogPersistedObject;
import org.encog.persist.Persistor;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.NumberList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The Encog persistor used to persist the ContextLayer class.
 * 
 * @author jheaton
 */
public class ContextLayerPersistor implements Persistor {

	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * Load the specified Encog object from an XML reader.
	 * 
	 * @param in
	 *            The XML reader to use.
	 * @return The loaded object.
	 */
	public EncogPersistedObject load(final ReadXML in) {

		int neuronCount = 0;
		int x = 0;
		int y = 0;
		String threshold = null;
		String context = null;
		ActivationFunction activation = null;
		final String end = in.getTag().getName();
		double biasActivation = 1;

		while (in.readToTag()) {
			if (in.is(BasicLayerPersistor.TAG_ACTIVATION, true)) {
				in.readToTag();
				final String type = in.getTag().getName();
				final Persistor persistor = PersistorUtil.createPersistor(type);
				activation = (ActivationFunction) persistor.load(in);
			} else if (in.is(BasicLayerPersistor.PROPERTY_NEURONS, true)) {
				neuronCount = in.readIntToTag();
			} else if (in.is(BasicLayerPersistor.PROPERTY_X, true)) {
				x = in.readIntToTag();
			} else if (in.is(BasicLayerPersistor.PROPERTY_Y, true)) {
				y = in.readIntToTag();
			} else if (in.is(BasicLayerPersistor.PROPERTY_THRESHOLD, true)) {
				threshold = in.readTextToTag();
			} else if (in.is(BasicLayerPersistor.PROPERTY_BIAS_ACTIVATION, true)) {
				biasActivation = Double.parseDouble(in.readTextToTag());
			} else if (in.is(end, false)) {
				break;
			}
		}

		if (neuronCount > 0) {
			ContextLayer layer;

			layer = new ContextLayer(activation, neuronCount);

			if (context != null) {
				final double[] t = NumberList.fromList(CSVFormat.EG_FORMAT,
						context);

				for (int i = 0; i < t.length; i++) {
					layer.getContext().setData(i, t[i]);
				}
			}

			layer.setX(x);
			layer.setY(y);
			

			return layer;
		}
		return null;
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
		PersistorUtil.beginEncogObject(
				EncogPersistedCollection.TYPE_CONTEXT_LAYER, out, obj, false);
		final ContextLayer layer = (ContextLayer) obj;

		out.addProperty(BasicLayerPersistor.PROPERTY_NEURONS, layer
				.getNeuronCount());
		out.addProperty(BasicLayerPersistor.PROPERTY_X, layer.getX());
		out.addProperty(BasicLayerPersistor.PROPERTY_Y, layer.getY());

		if (layer.hasBias()) {
			final StringBuilder result = new StringBuilder();
			NumberList
					.toList(CSVFormat.EG_FORMAT, result, layer.getBiasWeights());
			out.addProperty(BasicLayerPersistor.PROPERTY_THRESHOLD, result
					.toString());			
		}

		out.addProperty(BasicLayerPersistor.PROPERTY_BIAS_ACTIVATION, layer.getBiasActivation());
		out.beginTag(BasicLayerPersistor.TAG_ACTIVATION);
		final Persistor persistor = layer.getActivationFunction()
				.createPersistor();
		persistor.save(layer.getActivationFunction(), out);
		out.endTag();

		out.endTag();
	}

}
