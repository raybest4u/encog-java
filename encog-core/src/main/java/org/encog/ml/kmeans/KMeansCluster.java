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
package org.encog.ml.kmeans;

import java.util.ArrayList;
import java.util.List;

import org.encog.engine.data.BasicEngineDataSet;
import org.encog.engine.data.EngineData;
import org.encog.engine.data.EngineDataSet;
import org.encog.ml.MLCluster;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;

/**
 * 
 * @author jheaton
 * 
 */
public class KMeansCluster implements MLCluster {

	/**
	 * The centroid.
	 */
	private Centroid centroid;
	private double sumSqr;
	private final List<NeuralData> data = new ArrayList<NeuralData>();

	public void add(final NeuralData pair) { // called from CAInstance
		this.data.add(pair);
		calcSumOfSquares();
	}

	public void calcSumOfSquares() { // called from Centroid
		final int size = this.data.size();
		double temp = 0;
		for (int i = 0; i < size; i++) {
			temp += KMeansClustering.calculateEuclideanDistance(this.centroid,
					(this.data.get(i)));
		}
		this.sumSqr = temp;
	}

	public NeuralDataSet createDataSet() {
		final NeuralDataSet result = new BasicNeuralDataSet();

		for (final NeuralData data : this.data) {
			result.add(data);
		}

		return result;
	}

	public NeuralData get(final int pos) {
		return this.data.get(pos);
	}

	public Centroid getCentroid() {
		return this.centroid;
	}

	public List<NeuralData> getData() {
		return this.data;
	}

	public double getSumSqr() {
		return this.sumSqr;
	}

	public void remove(final NeuralData pair) {
		this.data.remove(pair);
		calcSumOfSquares();
	}

	public void setCentroid(final Centroid c) {
		this.centroid = c;
	}

	public int size() {
		return this.data.size();
	}

}
