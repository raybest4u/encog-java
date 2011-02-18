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
package org.encog.examples.neural.lunar;

import org.encog.app.quant.normalize.NormalizationAction;
import org.encog.app.quant.normalize.NormalizedField;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.neural.networks.BasicNetwork;

public class NeuralPilot {
	
	private BasicNetwork network;
	private boolean track;
    private NormalizedField fuelStats;
    private NormalizedField altitudeStats;
    private NormalizedField velocityStats;
	
	public NeuralPilot(BasicNetwork network, boolean track)
	{
        fuelStats = new NormalizedField(NormalizationAction.Normalize, "fuel", 200, 0, -0.9, 0.9);
        altitudeStats = new NormalizedField(NormalizationAction.Normalize, "altitude", 10000, 0, -0.9, 0.9);
        velocityStats = new NormalizedField(NormalizationAction.Normalize, "velocity", LanderSimulator.TERMINAL_VELOCITY, -LanderSimulator.TERMINAL_VELOCITY, -0.9, 0.9);

		this.track = track;
		this.network = network;
	}
	
	public int scorePilot()
	{
		LanderSimulator sim = new LanderSimulator();
		while(sim.flying())
		{
			NeuralData input = new BasicNeuralData(3);
            input.setData(0, this.fuelStats.normalize(sim.getFuel()));
            input.setData(1, this.fuelStats.normalize(sim.getAltitude()));
            input.setData(2, this.fuelStats.normalize(sim.getVelocity()));
            NeuralData output = this.network.compute(input);
            double value = output.getData(0);

            boolean thrust;
			
			if( value > 0 )
			{
				thrust = true;
				if( track )
					System.out.println("THRUST");
			}
			else
				thrust = false;
			
			sim.turn(thrust);
			if( track )
				System.out.println(sim.telemetry());
		}
		return(sim.score());
	}
}
