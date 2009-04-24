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
package org.encog.persist.persistors.generic;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.encog.EncogError;
import org.encog.neural.networks.layers.Layer;
import org.encog.parse.tags.read.ReadXML;
import org.encog.persist.EncogPersistedCollection;
import org.encog.persist.EncogPersistedObject;
import org.encog.persist.Persistor;
import org.encog.util.xml.XMLUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/**
 * A generic class used to take an XML segment and produce an object for it.
 * Some of the Encog persistors make use of this class.
 * @author jheaton
 *
 */
public class XML2Object {
	
	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());

	public void load(ReadXML in, EncogPersistedObject target) {

		try {
			target.setName(in.getTag().getAttributeValue("name"));
			target.setDescription(in.getTag().getAttributeValue("description"));
			
			while( in.readToTag() )
			{			
				String tagName = in.getTag().getName();
				Field field = target.getClass().getDeclaredField(
						tagName);
				field.setAccessible(true);
				String value =in.readTextToTag();
				
				Class<?> type = field.getType();
				if (type == long.class) {
					field.setLong(target, Long.parseLong(value));
				} else if (type == int.class) {
					field.setInt(target, Integer.parseInt(value));
				} else if (type == short.class) {
					field.setShort(target, Short.parseShort(value));
				} else if (type == String.class) {
					field.set(target, value);
				} else if (type == List.class) {
					field.set(target, loadList(in));
				}

			}
		} catch (NoSuchFieldException e) {
			throw new EncogError(e);
		} catch (NumberFormatException e) {
			throw new EncogError(e);
		} catch (IllegalArgumentException e) {
			throw new EncogError(e);
		} catch (IllegalAccessException e) {
			throw new EncogError(e);
		}
	}

	public List<Object> loadList(ReadXML in) {
		List<Object> result = new ArrayList<Object>();

		while( in.readToTag() )
		{			
			String tagName = in.getTag().getName();
			String value =in.readTextToTag(); 

			if (tagName.equals("S")) {
				result.add(value);
			}
		}

		return result;
	}

	public void loadObject(ReadXML in, EncogPersistedObject obj) {
		while( in.readToTag() )
		{			
			String tagName = in.getTag().getName();
			
			if( tagName.equals(obj.getClass().getSimpleName()))
			{
				load(in,obj);
				return;
			}
		}
	}
}
