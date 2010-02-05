/*
 * Encog(tm) Artificial Intelligence Framework v2.3
 * Java Version
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

package org.encog.util.math;

/**
 * Java will sometimes return Math.NaN or Math.Infinity when numbers get to
 * large or too small. This can have undesirable effects. This class provides
 * some basic math functions that may be in danger of returning such a value.
 * This class imposes a very large and small ceiling and floor to keep the
 * numbers within range.
 * 
 * @author jheaton
 * 
 */
public final class BoundMath {

	/**
	 * Calculate the cos.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double cos(final double a) {
		return BoundNumbers.bound(Math.cos(a));
	}

	/**
	 * Calculate the exp.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double exp(final double a) {
		return BoundNumbers.bound(Math.exp(a));
	}

	/**
	 * Calculate the log.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double log(final double a) {
		return BoundNumbers.bound(Math.log(a));
	}

	/**
	 * Calculate the power of a number.
	 * 
	 * @param a
	 *            The base.
	 * @param b
	 *            The exponent.
	 * @return The result of the function.
	 */
	public static double pow(final double a, final double b) {
		return BoundNumbers.bound(Math.pow(a, b));
	}

	/**
	 * Calculate the sin.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double sin(final double a) {
		return BoundNumbers.bound(Math.sin(a));
	}

	/**
	 * Calculate the square root.
	 * 
	 * @param a
	 *            The value passed to the function.
	 * @return The result of the function.
	 */
	public static double sqrt(final double a) {
		return Math.sqrt(a);
	}

	/**
	 * Private constructor.
	 */
	private BoundMath() {

	}
}
