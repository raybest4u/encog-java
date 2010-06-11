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

package org.encog.solve.genetic.species;

import java.util.List;

import org.encog.solve.genetic.genome.Genome;

/**
 * Defines the features used in a species. A species is a group of genomes.
 */
public interface Species {


	/**
	 * Calculate the amount that a species will spawn.
	 */
	void calculateSpawnAmount();

	/**
	 * Choose a worthy parent for mating.
	 * 
	 * @return The parent genome.
	 */
	Genome chooseParent();

	/**
	 * @return The age of this species.
	 */
	int getAge();

	/**
	 * @return The best score for this species.
	 */
	double getBestScore();

	/**
	 * @return How many generations with no improvement.
	 */
	int getGensNoImprovement();

	/**
	 * @return Get the leader for this species. The leader is the genome with
	 *         the best score.
	 */
	Genome getLeader();

	/**
	 * @return The numbers of this species.
	 */
	List<Genome> getMembers();

	/**
	 * @return The number of genomes this species will try to spawn into the
	 *         next generation.
	 */
	double getNumToSpawn();

	/**
	 * @return The number of spawns this species requires.
	 */
	double getSpawnsRequired();

	/**
	 * @return The species ID.
	 */
	long getSpeciesID();

	/**
	 * Purge old unsuccessful genomes.
	 */
	void purge();

	/**
	 * Set the age of this species.
	 * @param age The age.
	 */
	void setAge(int age);

	/**
	 * Set the best score.
	 * @param bestScore The best score.
	 */
	void setBestScore(double bestScore);

	/**
	 * Set the number of generations with no improvement.
	 * @param gensNoImprovement The number of generations with
	 * no improvement.
	 */
	void setGensNoImprovement(int gensNoImprovement);

	/**
	 * Set the leader of this species.
	 * @param leader The leader of this species.
	 */
	void setLeader(Genome leader);

	/**
	 * Set the number of spawns required.
	 * @param spawnsRequired The number of spawns required.
	 */
	void setSpawnsRequired(double spawnsRequired);
}
