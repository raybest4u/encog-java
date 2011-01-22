/*
 * Encog(tm) Core v3.0 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package org.encog.bot.dataunit;

import org.encog.parse.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A data unit that holds a tag.
 *
 * @author jheaton
 *
 */
public class TagDataUnit extends DataUnit {

	/**
	 * The tag for this data unit.
	 */
	private Tag tag;

	/**
	 * The logger.
	 */
	@SuppressWarnings("unused")
	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * @return The tag that this data unit is based on.
	 */
	public Tag getTag() {
		return this.tag;
	}

	/**
	 * Set the tag that this data unit is based on.
	 *
	 * @param tag
	 *            HTML tag.
	 */
	public void setTag(final Tag tag) {
		this.tag = tag;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String toString() {
		return this.tag.toString();
	}
}
