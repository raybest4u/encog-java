/*
 * Encog(tm) Core v2.6 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2010 Heaton Research, Inc.
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
package org.encog.engine.cluster.kmeans;

/**
 * The centers of each cluster.
 */
public class Centroid {
	/**
	 * The center for each dimension in the input.
	 */
	private final double[] centers;
	
	/**
	 * The cluster.
	 */
	private KMeansCluster cluster;

	/**
	 * Construct the centroid.
	 * @param centers The centers.
	 */
	public Centroid(final double[] centers) {
		this.centers = centers;
	}

	/**
	 * Calculate the centroid.
	 */
	public void calcCentroid() { // only called by CAInstance
		final int numDP = this.cluster.size();

		final double[] temp = new double[this.centers.length];

		// caluclating the new Centroid
		for (int i = 0; i < numDP; i++) {
			for (int j = 0; j < temp.length; j++) {
				temp[j] += this.cluster.get(i).getInputArray()[j];
			}
		}

		for (int i = 0; i < temp.length; i++) {
			this.centers[i] = temp[i] / numDP;
		}

		this.cluster.calcSumOfSquares();
	}

	/**
	 * @return The centers.
	 */
	public double[] getCenters() {
		return this.centers;
	}

	/**
	 * @return The clusters.
	 */
	public KMeansCluster getCluster() {
		return this.cluster;
	}

	/**
	 * Set the cluster.
	 * @param c The cluster.
	 */
	public void setCluster(final KMeansCluster c) {
		this.cluster = c;
	}

}
