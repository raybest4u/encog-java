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
package org.encog.bot.browse.extract;

import java.util.Collection;
import java.util.List;

import org.encog.bot.browse.WebPage;

/**
 * Provides the basic interface that any extractor must support.  An 
 * extractor is a class that is capable of extracting certain types of
 * data from web data.  For example, the ExtractWords extractor is used
 * to extract all of the words from a web page.
 * @author jheaton
 *
 */
public interface Extract {
	public void extract(WebPage page);
	public List<Object> extractList(WebPage page);
	public void addListener(ExtractListener listener);
	public void removeListener(ExtractListener listener);
	public Collection<ExtractListener> getListeners();
}
