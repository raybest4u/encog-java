package org.encog.bot;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL; 

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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for bots.
 * @author jheaton
 *
 */
public final class BotUtil {

	/**
	 * Private constructor.
	 */
	private BotUtil() {
		
	}
	
	/**
	 * How much data to read at once.
	 */
	public static final int BUFFER_SIZE = 8192;

	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(BotUtil.class);

	/**
	 * Load the specified web page into a string.
	 * @param url The url to load.
	 * @return The web page as a string.
	 */
	public static String loadPage(final URL url) {
		try {
			StringBuilder result = new StringBuilder();
			byte[] buffer = new byte[BUFFER_SIZE];

			int length;

			InputStream is = url.openStream();
			
			do {
				length = is.read(buffer);
				if (length >= 0) {
					result.append(new String(buffer, 0, length));
				}
			} while (length >= 0);

			return result.toString();
		} catch (IOException e) {
			if (BotUtil.LOGGER.isErrorEnabled()) {
				BotUtil.LOGGER.error("Exception", e);
			}
			throw new BotError(e);
		}
	}
}
