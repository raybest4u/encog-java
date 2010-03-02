/*
 * Encog(tm) Core v2.4
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

package org.encog.util;

/**
 * Some array functions used by Encog.
 */
public final class EncogArray {

	/**
	 * Completely copy one array into another.
	 * 
	 * @param src
	 *            Source array.
	 * @param dst
	 *            Destination array.
	 */
	public static void arrayCopy(final double[] src, final double[] dst) {
		System.arraycopy(src, 0, dst, 0, src.length);
	}

	/**
	 * Convert an array of double primitives to Double objects.
	 * @param array The primitive array.
	 * @return The object array.
	 */
	public static Double[] doubleToObject(final double[] array) {
		final Double[] result = new Double[array.length];
		for (int i = 0; i < array.length; i++) {
			result[i] = new Double(array[i]);
		}
		return result;
	}

	/**
	 * Convert an array of Double objects to double primitives.
	 * @param array An array of Double objects.
	 * @return An array of double primitives.
	 */
	public static double[] objectToDouble(final Double[] array) {
		final double[] result = new double[array.length];
		for (int i = 0; i < array.length; i++) {
			result[i] = new Double(array[i]);
		}
		return result;
	}

	/**
	 * Calculate the product of two vectors (a scalar value).
	 * 
	 * @param a
	 *            First vector to multiply.
	 * @param b
	 *            Second vector to multiply.
	 * @return The product of the two vectors (a scalar value).
	 */
	public static double vectorProduct(final double[] a, 
			final double[] b) {
		final int length = a.length;
		double value = 0;

		for (int i = 0; i < length; ++i) {
			value += a[i] * b[i];
		}

		return value;
	}

	/**
	 * Private constructor.
	 */
	private EncogArray() {

	}
}
