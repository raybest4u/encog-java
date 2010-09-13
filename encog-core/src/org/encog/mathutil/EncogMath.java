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

package org.encog.mathutil;

/**
 * Several useful math functions for Encog.
 */
public final class EncogMath {
	
	
	/**
	 * Private constructor.
	 */
	private EncogMath() {
		
	}
	
	/**
	 * Convert degrees to radians.
	 * 
	 * @param deg
	 *            Degrees.
	 * @return Radians.
	 */
	public static double deg2rad(final double deg) {
		return deg * (Math.PI / MathConst.DEG_SEMICIRCLE);
	}

	/**
	 * sqrt(a^2 + b^2) without under/overflow.
	 * @param a First param.
	 * @param b Second param.
	 * @return The result.
	 */
	public static double hypot(final double a, final double b) {
		double r;
		if (Math.abs(a) > Math.abs(b)) {
			r = b / a;
			r = Math.abs(a) * Math.sqrt(1 + r * r);
		} else if (b != 0) {
			r = a / b;
			r = Math.abs(b) * Math.sqrt(1 + r * r);
		} else {
			r = 0.0;
		}
		return r;
	}

	/**
	 * Convert radians to degrees.
	 * 
	 * @param rad
	 *            Radians
	 * @return Degrees.
	 */
	public static double rad2deg(final double rad) {
		return rad * (MathConst.DEG_SEMICIRCLE / Math.PI);
	}
}
