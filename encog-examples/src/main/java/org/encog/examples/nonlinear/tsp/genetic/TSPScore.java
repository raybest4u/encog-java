/*
 * Encog(tm) Examples v3.0 - Java Version
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
package org.encog.examples.nonlinear.tsp.genetic;

import org.encog.examples.nonlinear.tsp.City;
import org.encog.ml.genetic.genome.CalculateGenomeScore;
import org.encog.ml.genetic.genome.Genome;

public class TSPScore implements CalculateGenomeScore {

	private City[] cities;
	
	public TSPScore(City[] cities)
	{
		this.cities = cities;
	}
	
	@Override
	public double calculateScore(Genome genome) {
		double result = 0.0;
		
		int[] path = (int[])genome.getOrganism();
		
		for (int i = 0; i < cities.length - 1; i++) {
			City city1 = cities[path[i]];
			City city2 = cities[path[i+1]];
			
			final double dist = city1.proximity(city2);
			result += dist;
		}
		
		return result;
	}

	public boolean shouldMinimize() {
		return true;
	}

}
