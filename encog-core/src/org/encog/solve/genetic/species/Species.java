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
package org.encog.solve.genetic.species;

import java.util.List;

import org.encog.neural.networks.training.neat.NEATGenome;
import org.encog.solve.genetic.genome.Genome;

/**
 * Defines the features used in a species. A species is a group of genomes.
 */
public interface Species {

	void addMember(NEATGenome genome);

	void adjustFitness();

	void calculateSpawnAmount();

	Genome chooseParent();

	int getAge();

	double getBestFitness();

	int getGensNoImprovement();

	Genome getLeader();

	List<Genome> getMembers();

	double getNumToSpawn();

	double getSpawnsRequired();

	long getSpeciesID();

	void purge();

	void setAge(int age);

	void setBestFitness(double bestFitness);

	void setGensNoImprovement(int gensNoImprovement);

	void setLeader(NEATGenome leader);

	void setSpawnsRequired(double spawnsRequired);
}
