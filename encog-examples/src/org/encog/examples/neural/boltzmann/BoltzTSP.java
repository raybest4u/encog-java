/*
 * Encog(tm) Examples v2.6 
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
package org.encog.examples.neural.boltzmann;

import org.encog.neural.data.bipolar.BiPolarNeuralData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.logic.BoltzmannLogic;
import org.encog.neural.pattern.BoltzmannPattern;

public class BoltzTSP {

	public static final int NUM_CITIES = 10;
	public static final int NEURON_COUNT = NUM_CITIES * NUM_CITIES;

	private double gamma = 7;
	private double[][] distance;		

	public double sqr(double x) {
		return x * x;
	}

	public void createCities() {
		double x1, x2, y1, y2;
		double alpha1, alpha2;
		
		this.distance = new double[NUM_CITIES][NUM_CITIES];

		for (int n1 = 0; n1 < NUM_CITIES; n1++) {
			for (int n2 = 0; n2 < NUM_CITIES; n2++) {
				alpha1 = ((double) n1 / NUM_CITIES) * 2 * Math.PI;
				alpha2 = ((double) n2 / NUM_CITIES) * 2 * Math.PI;
				x1 = Math.cos(alpha1);
				y1 = Math.sin(alpha1);
				x2 = Math.cos(alpha2);
				y2 = Math.sin(alpha2);
				distance[n1][n2] = Math.sqrt(sqr(x1 - x2) + sqr(y1 - y2));
			}
		}		
	}

	public boolean isValidTour(BiPolarNeuralData data) {
		int cities, stops;

		for (int n1 = 0; n1 < NUM_CITIES; n1++) {
			cities = 0;
			stops = 0;
			for (int n2 = 0; n2 < NUM_CITIES; n2++) {
				if (data.getBoolean(n1 * NUM_CITIES + n2)) {
					if (++cities > 1)
						return false;
				}
				if (data.getBoolean(n2 * NUM_CITIES + n1)) {
					if (++stops > 1)
						return false;
				}
			}
			if ((cities != 1) || (stops != 1))
				return false;
		}
		return true;
	}

	public double lengthOfTour(BiPolarNeuralData data) {
		double result;
		int n1, n2, n3;

		result = 0;
		for (n1 = 0; n1 < NUM_CITIES; n1++) {
			for (n2 = 0; n2 < NUM_CITIES; n2++) {
				if (data.getBoolean(((n1) % NUM_CITIES) * NUM_CITIES + n2))
					break;
			}
			for (n3 = 0; n3 < NUM_CITIES; n3++) {
				if (data.getBoolean(((n1 + 1) % NUM_CITIES) * NUM_CITIES + n3))
					break;
			}
			result += distance[n2][n3];
		}
		return result;
	}

	String displayTour(BiPolarNeuralData data) {
		StringBuilder result = new StringBuilder();

		int n1, n2;
		boolean first;

		for (n1 = 0; n1 < NUM_CITIES; n1++) {
			first = true;
			result.append("[");
			for (n2 = 0; n2 < NUM_CITIES; n2++) {
				if (data.getBoolean(n1 * NUM_CITIES + n2)) {
					if (first) {
						first = false;
						result.append(n2);
					} else {
						result.append(", " + n2);
					}
				}
			}
			result.append("]");
			if (n1 != NUM_CITIES - 1) {
				result.append(" -> ");
			}
		}
		return result.toString();
	}

	public void calculateWeights(BasicNetwork network) {
		
		BoltzmannLogic logic = (BoltzmannLogic)network.getLogic();

		for (int sourceTour = 0; sourceTour < NUM_CITIES; sourceTour++) {
			for (int sourceCity = 0; sourceCity < NUM_CITIES; sourceCity++) {
				int sourceIndex = sourceTour * NUM_CITIES + sourceCity;
				for (int targetTour = 0; targetTour < NUM_CITIES; targetTour++) {
					for (int targetCity = 0; targetCity < NUM_CITIES; targetCity++) {
						int targetIndex = targetTour * NUM_CITIES + targetCity;
						double weight = 0;
						if (sourceIndex != targetIndex) {
							int predTargetTour = (targetTour == 0 ? NUM_CITIES - 1 : targetTour - 1);
							int succTargetTour = (targetTour == NUM_CITIES - 1 ? 0 : targetTour + 1);
							if ((sourceTour == targetTour) || (sourceCity == targetCity))
								weight = -gamma;
							else if ((sourceTour == predTargetTour) || (sourceTour == succTargetTour))
								weight = -distance[sourceCity][targetCity];
						}
						logic.getThermalSynapse().getMatrix().set(sourceIndex, targetIndex, weight);
					}
				}
				logic.getThermalLayer().setBiasWeight(sourceIndex, -gamma / 2);
			}
		}
	}


	public void run() {
		BoltzmannPattern pattern = new BoltzmannPattern();
		pattern.setInputNeurons(NEURON_COUNT);
		BasicNetwork network = pattern.generate();
		BoltzmannLogic logic = (BoltzmannLogic)network.getLogic();

		createCities();
		calculateWeights(network);

		logic.setTemperature(100);
		do {
			logic.establishEquilibrium();
			System.out.println(logic.getTemperature()+" : "+displayTour(logic.getCurrentState()));
			logic.decreaseTemperature(0.99);
		} while (!isValidTour(logic.getCurrentState()));

		System.out.println("Final Length: " + this.lengthOfTour(logic.getCurrentState()) );
	}

	public static void main(String[] args) {
		BoltzTSP program = new BoltzTSP();
		program.run();
	}

}
