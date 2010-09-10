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

package org.encog.neural.networks.training.propagation;

import java.util.HashMap;
import java.util.Map;

import org.encog.persist.BasicPersistedObject;
import org.encog.persist.Persistor;
import org.encog.persist.persistors.TrainingContinuationPersistor;

/**
 * Allows training to be continued.
 * 
 */
public class TrainingContinuation extends BasicPersistedObject {

	/**
	 * The serial ID.
	 */
	private static final long serialVersionUID = -3649790586015301015L;
			
	/**
	 * The contents of this object.
	 */
	private final Map<String, Object> contents = new HashMap<String, Object>();

	/**
	 * @return A persistor for this object.
	 */
	public Persistor createPersistor() {
		return new TrainingContinuationPersistor();
	}

	/**
	 * Get an object by name.
	 * @param name The name of the object.
	 * @return The object requested.
	 */
	public Object get(final String name) {
		return this.contents.get(name);
	}

	/**
	 * @return The contents.
	 */
	public Map<String, Object> getContents() {
		return this.contents;
	}

	/**
	 * Save a list of doubles.
	 * @param key The key to save them under.
	 * @param list The list of doubles.
	 */
	public void put(final String key, final double[] list) {
		this.contents.put(key, list);
	}

	/**
	 * Set a value to a string.
	 * @param name The value to set.
	 * @param value The value.
	 */
	public void set(final String name, final Object value) {
		this.contents.put(name, value);
	}
}
