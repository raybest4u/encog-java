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

package org.encog.neural.activation;

import junit.framework.TestCase;

import org.junit.Assert;
import org.junit.Test;

public class TestActivationGaussian extends TestCase {
	
	@Test
	public void testGaussian() throws Throwable
	{
		ActivationGaussian activation = new ActivationGaussian(0.0,0.5,1.0);
		Assert.assertTrue(activation.hasDerivative());
		
		ActivationGaussian clone = (ActivationGaussian)activation.clone();
		Assert.assertNotNull(clone);
		
		double[] input = { 0.0  };
		
		activation.activationFunction(input);
		
		Assert.assertEquals(0.5,input[0],0.1);
	
		
		// test derivative, should throw an error
		
		activation.derivativeFunction(input);
		Assert.assertEquals(-33,(int)(input[0]*100),0.1);		
		
		
		
	}
}
