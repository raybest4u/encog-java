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

package org.encog.neural.networks.training.neat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.encog.mathutil.randomize.RangeRandomizer;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.activation.ActivationLinear;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.synapse.neat.NEATLink;
import org.encog.neural.networks.synapse.neat.NEATNeuron;
import org.encog.neural.networks.synapse.neat.NEATNeuronType;
import org.encog.neural.networks.synapse.neat.NEATSynapse;
import org.encog.neural.pattern.NEATPattern;
import org.encog.persist.annotations.EGAttribute;
import org.encog.persist.annotations.EGReference;
import org.encog.solve.genetic.genes.Gene;
import org.encog.solve.genetic.genome.BasicGenome;
import org.encog.solve.genetic.genome.Chromosome;

/**
 * Implements a NEAT genome.  This is a "blueprint" for creating a neural network.
 * 
 * NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm for the
 * generation of evolving artificial neural networks. It was developed by Ken
 * Stanley while at The University of Texas at Austin.
 * 
 * http://www.cs.ucf.edu/~kstanley/
 * 
 */
public class NEATGenome extends BasicGenome implements Cloneable {

	/**
	 * The adjustment factor for disjoint genes.
	 */
	public static final double TWEAK_DISJOINT = 1;
	
	/**
	 * The adjustment factor for excess genes.
	 */
	public static final double TWEAK_EXCESS = 1;
	
	/**
	 * The adjustment factor for matched genes.
	 */
	public static final double TWEAK_MATCHED = 0.4;

	/**
	 * The number of inputs.
	 */
	@EGAttribute
	private int inputCount;
	
	/**
	 * The chromsome that holds the links.
	 */
	@EGReference
	private Chromosome linksChromosome;
	
	/**
	 * THe network depth.
	 */
	@EGAttribute
	private int networkDepth;
	
	/**
	 * The chromosome that holds the neurons.
	 */
	@EGReference
	private Chromosome neuronsChromosome;
	
	/**
	 * The number of outputs.
	 */
	@EGAttribute
	private int outputCount;
	
	/**
	 * The species id.
	 */
	@EGAttribute
	private long speciesID;
	

	public NEATGenome()
	{
		super(null);
	}
	
	
	/**
	 * Construct a genome by copying another.
	 * @param other The other genome.
	 */
	public NEATGenome(final NEATGenome other) {
		super(other.getGeneticAlgorithm());

		neuronsChromosome = new Chromosome();
		linksChromosome = new Chromosome();
		
		getChromosomes().add(neuronsChromosome);
		getChromosomes().add(linksChromosome);

		setGenomeID(other.getGenomeID());
		networkDepth = other.networkDepth;
		setScore(other.getScore());
		setAdjustedScore(other.getAdjustedScore());
		setAmountToSpawn(other.getAmountToSpawn());
		inputCount = other.inputCount;
		outputCount = other.outputCount;
		speciesID = other.speciesID;

		// copy neurons
		for (final Gene gene : other.getNeurons().getGenes()) {
			final NEATNeuronGene oldGene = (NEATNeuronGene) gene;
			final NEATNeuronGene newGene = new NEATNeuronGene(oldGene
					.getNeuronType(), oldGene.getId(), oldGene.getSplitY(),
					oldGene.getSplitX(), oldGene.isRecurrent(), oldGene
							.getActivationResponse());
			getNeurons().add(newGene);
		}

		// copy links
		for (final Gene gene : other.getLinks().getGenes()) {
			final NEATLinkGene oldGene = (NEATLinkGene) gene;
			final NEATLinkGene newGene = new NEATLinkGene(oldGene
					.getFromNeuronID(), oldGene.getToNeuronID(), oldGene
					.isEnabled(), oldGene.getInnovationId(), oldGene
					.getWeight(), oldGene.isRecurrent());
			getLinks().add(newGene);
		}

	}

	/**
	 * Create a NEAT gnome.
	 * @param training The owner object.
	 * @param genomeID The genome id.
	 * @param neurons The neurons.
	 * @param links The links.
	 * @param inputCount The input count.
	 * @param outputCount The output count.
	 */
	public NEATGenome(final NEATTraining training, final long genomeID,
			final Chromosome neurons, final Chromosome links,
			final int inputCount, final int outputCount) {
		super(training);
		setGenomeID(genomeID);
		linksChromosome = links;
		neuronsChromosome = neurons;
		setAmountToSpawn(0);
		setAdjustedScore(0);
		this.inputCount = inputCount;
		this.outputCount = outputCount;
		
		getChromosomes().add(neuronsChromosome);
		getChromosomes().add(linksChromosome);
	}

	/**
	 * Construct a genome, do not provide links and neurons.
	 * @param training The owner object.
	 * @param id The genome id.
	 * @param inputCount The input count.
	 * @param outputCount The output count.
	 */
	public NEATGenome(final NEATTraining training, final long id,
			final int inputCount, final int outputCount) {
		super(training);
		setGenomeID(id);
		setAdjustedScore(0);
		this.inputCount = inputCount;
		this.outputCount = outputCount;
		setAmountToSpawn(0);
		speciesID = 0;

		final double inputRowSlice = 0.8 / (inputCount);
		neuronsChromosome = new Chromosome();
		linksChromosome = new Chromosome();
		
		getChromosomes().add(neuronsChromosome);
		getChromosomes().add(linksChromosome);

		for (int i = 0; i < inputCount; i++) {
			neuronsChromosome.add(new NEATNeuronGene(NEATNeuronType.Input, i,
					0, 0.1 + i * inputRowSlice));
		}

		neuronsChromosome.add(new NEATNeuronGene(NEATNeuronType.Bias,
				inputCount, 0, 0.9));

		final double outputRowSlice = 1 / (double) (outputCount + 1);

		for (int i = 0; i < outputCount; i++) {
			neuronsChromosome.add(new NEATNeuronGene(NEATNeuronType.Output, i
					+ inputCount + 1, 1, (i + 1) * outputRowSlice));
		}

		for (int i = 0; i < inputCount + 1; i++) {
			for (int j = 0; j < outputCount; j++) {
				linksChromosome.add(new NEATLinkGene(
						((NEATNeuronGene) neuronsChromosome.get(i)).getId(),
						((NEATNeuronGene) getNeurons().get(inputCount + j + 1))
								.getId(), true, inputCount + outputCount + 1
								+ getNumGenes(), RangeRandomizer.randomize(-1,
								1), false));
			}
		}

	}

	/**
	 * Mutate the genome by adding a link to this genome.
	 * @param mutationRate The mutation rate.
	 * @param chanceOfLooped The chance of a self-connected neuron.
	 * @param numTrysToFindLoop The number of tries to find a loop.
	 * @param numTrysToAddLink The number of tries to add a link.
	 */
	void addLink(final double mutationRate, final double chanceOfLooped,
			int numTrysToFindLoop, int numTrysToAddLink) {

		// should we even add the link
		if (Math.random() > mutationRate) {
			return;
		}

		// the link will be between these two neurons
		long neuron1ID = -1;
		long neuron2ID = -1;

		boolean recurrent = false;

		// a self-connected loop?
		if (Math.random() < chanceOfLooped) {

			// try to find(randomly) a neuron to add a self-connected link to
			while ((numTrysToFindLoop--) > 0) {
				final NEATNeuronGene neuronGene = chooseRandomNeuron(false);

				// no self-links on input or bias neurons
				if (!neuronGene.isRecurrent()
						&& (neuronGene.getNeuronType() != NEATNeuronType.Bias)
						&& (neuronGene.getNeuronType() != NEATNeuronType.Input)) {
					neuron1ID = neuron2ID = neuronGene.getId();

					neuronGene.setRecurrent(true);
					recurrent = true;

					numTrysToFindLoop = 0;
				}
			}
		} else {
			// try to add a regular link
			while ((numTrysToAddLink--) > 0) {
				final NEATNeuronGene neuron1 = chooseRandomNeuron(true);
				final NEATNeuronGene neuron2 = chooseRandomNeuron(false);

				if (!isDuplicateLink(neuron1ID, neuron2ID)
						&& (neuron1.getId() != neuron2.getId())
						&& (neuron2.getNeuronType() != NEATNeuronType.Bias)) {

					neuron1ID = neuron1.getId();
					neuron2ID = neuron2.getId();
					break;
				}
			}
		}

		// did we fail to find a link
		if ((neuron1ID < 0) || (neuron2ID < 0)) {
			return;
		}

		// check to see if this innovation has already been tried
		final NEATInnovation innovation = ((NEATTraining)getGeneticAlgorithm())
			.getInnovations()
			.checkInnovation(neuron1ID, neuron1ID,
				NEATInnovationType.NewLink);

		// see if this is a recurrent(backwards) link
		final NEATNeuronGene neuronGene = (NEATNeuronGene) neuronsChromosome
				.get(getElementPos(neuron1ID));
		if (neuronGene.getSplitY() > neuronGene.getSplitY()) {
			recurrent = true;
		}

		// is this a new innovation?
		if (innovation == null) {
			// new innovation
			((NEATTraining)getGeneticAlgorithm()).getInnovations().createNewInnovation(neuron1ID, neuron2ID,
					NEATInnovationType.NewLink);

			final long id2 = ((NEATTraining)getGeneticAlgorithm()).getPopulation().assignInnovationID();

			final NEATLinkGene linkGene = new NEATLinkGene(neuron1ID,
					neuron2ID, true, id2, RangeRandomizer.randomize(-1, 1),
					recurrent);
			linksChromosome.add(linkGene);
		} else {
			// existing innovation
			final NEATLinkGene linkGene = new NEATLinkGene(neuron1ID,
					neuron2ID, true, innovation.getInnovationID(),
					RangeRandomizer.randomize(-1, 1), recurrent);
			linksChromosome.add(linkGene);
		}
	}

	
	/**
	 * Mutate the genome by adding a neuron.
	 * @param mutationRate The mutation rate.
	 * @param numTrysToFindOldLink The number of tries to find a link to split.
	 */
	void addNeuron(final double mutationRate, int numTrysToFindOldLink) {

		// should we add a neuron?
		if (Math.random() > mutationRate) {
			return;
		}

		// the link to split
		NEATLinkGene splitLink = null;

		final int sizeThreshold = inputCount + outputCount + 10;

		// if there are not at least
		int upperLimit;
		if (linksChromosome.size() < sizeThreshold) {
			upperLimit = getNumGenes() - 1 - (int) Math.sqrt(getNumGenes());
		} else {
			upperLimit = getNumGenes() - 1;
		}

		while ((numTrysToFindOldLink--) > 0) {
			// choose a link, use the square root to prefer the older links
			final int i = RangeRandomizer.randomInt(0, upperLimit);
			final NEATLinkGene link = (NEATLinkGene) linksChromosome.get(i);

			// get the from neuron
			final long fromNeuron = link.getFromNeuronID();

			if ((link.isEnabled())
					&& (!link.isRecurrent())
					&& (((NEATNeuronGene) getNeurons().get(
							getElementPos(fromNeuron))).getNeuronType() != NEATNeuronType.Bias)) {
				splitLink = link;
				break;
			}
		}

		if (splitLink == null) {
			return;
		}

		splitLink.setEnabled(false);

		final double originalWeight = splitLink.getWeight();

		final long from = splitLink.getFromNeuronID();
		final long to = splitLink.getToNeuronID();

		final NEATNeuronGene fromGene = (NEATNeuronGene) getNeurons().get(
				getElementPos(from));
		final NEATNeuronGene toGene = (NEATNeuronGene) getNeurons().get(
				getElementPos(to));

		final double newDepth = (fromGene.getSplitY() + toGene.getSplitY()) / 2;
		final double newWidth = (fromGene.getSplitX() + toGene.getSplitX()) / 2;

		// has this innovation already been tried?
		NEATInnovation innovation = ((NEATTraining)getGeneticAlgorithm()).getInnovations().checkInnovation(
				from, to, NEATInnovationType.NewNeuron);

		// prevent chaining
		if (innovation != null) {
			final long neuronID = innovation.getNeuronID();

			if (alreadyHaveThisNeuronID(neuronID)) {
				innovation = null;
			}
		}

		if (innovation == null) {
			// this innovation has not been tried, create it
			final long newNeuronID = ((NEATTraining)getGeneticAlgorithm()).getInnovations()
					.createNewInnovation(from, to,
							NEATInnovationType.NewNeuron,
							NEATNeuronType.Hidden, newWidth, newDepth);

			neuronsChromosome.add(new NEATNeuronGene(NEATNeuronType.Hidden,
					newNeuronID, newDepth, newWidth));

			// add the first link
			final long link1ID = ((NEATTraining)getGeneticAlgorithm()).getPopulation().assignInnovationID();

			((NEATTraining)getGeneticAlgorithm()).getInnovations().createNewInnovation(from, newNeuronID,
					NEATInnovationType.NewLink);

			final NEATLinkGene link1 = new NEATLinkGene(from, newNeuronID,
					true, link1ID, 1.0, false);

			linksChromosome.add(link1);

			// add the second link
			final long link2ID = ((NEATTraining)getGeneticAlgorithm()).getPopulation().assignInnovationID();

			((NEATTraining)getGeneticAlgorithm()).getInnovations().createNewInnovation(newNeuronID, to,
					NEATInnovationType.NewLink);

			final NEATLinkGene link2 = new NEATLinkGene(newNeuronID, to, true,
					link2ID, originalWeight, false);

			linksChromosome.add(link2);
		}

		else {
			// existing innovation
			final long newNeuronID = innovation.getNeuronID();

			final NEATInnovation innovationLink1 = ((NEATTraining)getGeneticAlgorithm()).getInnovations()
					.checkInnovation(from, newNeuronID,
							NEATInnovationType.NewLink);
			final NEATInnovation innovationLink2 = ((NEATTraining)getGeneticAlgorithm()).getInnovations()
					.checkInnovation(newNeuronID, to,
							NEATInnovationType.NewLink);

			if ((innovationLink1 == null) || (innovationLink2 == null)) {
				throw new NeuralNetworkError("NEAT Error");
			}

			final NEATLinkGene link1 = new NEATLinkGene(from, newNeuronID,
					true, innovationLink1.getInnovationID(), 1.0, false);
			final NEATLinkGene link2 = new NEATLinkGene(newNeuronID, to, true,
					innovationLink2.getInnovationID(), originalWeight, false);

			linksChromosome.add(link1);
			linksChromosome.add(link2);

			final NEATNeuronGene newNeuron = new NEATNeuronGene(
					NEATNeuronType.Hidden, newNeuronID, newDepth, newWidth);

			neuronsChromosome.add(newNeuron);
		}

		return;
	}

	/**
	 * Do we already have this neuron id?
	 * @param id The id to check for.
	 * @return True if we already have this neuron id.
	 */
	public boolean alreadyHaveThisNeuronID(final long id) {
		for (final Gene gene : neuronsChromosome.getGenes()) {

			final NEATNeuronGene neuronGene = (NEATNeuronGene) gene;

			if (neuronGene.getId() == id) {
				return true;
			}
		}

		return false;
	}

	/**
	 * Choose a random neuron.
	 * @param includeInput Should the input neurons be included.
	 * @return The random neuron.
	 */
	private NEATNeuronGene chooseRandomNeuron(final boolean includeInput) {
		int start;

		if (includeInput) {
			start = 0;
		} else {
			start = inputCount + 1;
		}

		final int neuronPos = RangeRandomizer.randomInt(start, getNeurons()
				.size() - 1);
		final NEATNeuronGene neuronGene = (NEATNeuronGene) neuronsChromosome
				.get(neuronPos);
		return neuronGene;

	}

	/**
	 * Convert the genes to an actual network.
	 */
	public void decode() {
		
		NEATPattern pattern = new NEATPattern();
		
		final List<NEATNeuron> neurons = pattern.getNeurons();

		for (final Gene gene : getNeurons().getGenes()) {
			final NEATNeuronGene neuronGene = (NEATNeuronGene) gene;
			final NEATNeuron neuron = new NEATNeuron(
					neuronGene.getNeuronType(), neuronGene.getId(), neuronGene
							.getSplitY(), neuronGene.getSplitX(), neuronGene
							.getActivationResponse());

			neurons.add(neuron);
		}

		// now to create the links.
		for (final Gene gene : getLinks().getGenes()) {
			final NEATLinkGene linkGene = (NEATLinkGene) gene;
			if (linkGene.isEnabled()) {
				int element = getElementPos(linkGene.getFromNeuronID());
				final NEATNeuron fromNeuron = neurons.get(element);

				element = getElementPos(linkGene.getToNeuronID());
				final NEATNeuron toNeuron = neurons.get(element);

				final NEATLink link = new NEATLink(linkGene.getWeight(),
						fromNeuron, toNeuron, linkGene.isRecurrent());

				fromNeuron.getOutputboundLinks().add(link);
				toNeuron.getInboundLinks().add(link);

			}
		}

		pattern.setNEATActivationFunction(((NEATTraining)getGeneticAlgorithm()).getNeatActivationFunction());
		pattern.setActivationFunction(((NEATTraining)getGeneticAlgorithm()).getOutputActivationFunction());
		pattern.setInputNeurons(inputCount);
		pattern.setOutputNeurons(outputCount);
		pattern.setSnapshot(((NEATTraining)getGeneticAlgorithm()).isSnapshot());
	
		setOrganism(pattern.generate());

	}

	/**
	 * Convert the network to genes.  Not currently supported.
	 */
	public void encode() {		

	}

	/**
	 * Get the compatibility score with another genome.  Used to determine species.
	 * @param genome The other genome.
	 * @return The score.
	 */
	public double getCompatibilityScore(final NEATGenome genome) {
		double numDisjoint = 0;
		double numExcess = 0;
		double numMatched = 0;
		double weightDifference = 0;

		int g1 = 0;
		int g2 = 0;

		while ((g1 < linksChromosome.size() - 1)
				|| (g2 < linksChromosome.size() - 1)) {

			if (g1 == linksChromosome.size() - 1) {
				g2++;
				numExcess++;

				continue;
			}

			if (g2 == genome.getLinks().size() - 1) {
				g1++;
				numExcess++;

				continue;
			}

			// get innovation numbers for each gene at this point
			final long id1 = ((NEATLinkGene) linksChromosome.get(g1))
					.getInnovationId();
			final long id2 = ((NEATLinkGene) genome.getLinks().get(g2))
					.getInnovationId();

			// innovation numbers are identical so increase the matched score
			if (id1 == id2) {
				g1++;
				g2++;
				numMatched++;

				// get the weight difference between these two genes
				weightDifference += Math.abs(((NEATLinkGene) linksChromosome
						.get(g1)).getWeight()
						- ((NEATLinkGene) genome.getLinks().get(g2))
								.getWeight());
			}

			// innovation numbers are different so increment the disjoint score
			if (id1 < id2) {
				numDisjoint++;
				g1++;
			}

			if (id1 > id2) {
				++numDisjoint;
				++g2;
			}

		}

		int longest = genome.getNumGenes();

		if (getNumGenes() > longest) {
			longest = getNumGenes();
		}

		final double score = (NEATGenome.TWEAK_EXCESS * numExcess / longest)
				+ (NEATGenome.TWEAK_DISJOINT * numDisjoint / longest)
				+ (NEATGenome.TWEAK_MATCHED * weightDifference / numMatched);

		return score;
	}

	/**
	 * Get the specified neuron's index.
	 * @param neuronID The neuron id to check for.
	 * @return The index.
	 */
	private int getElementPos(final long neuronID) {

		for (int i = 0; i < getNeurons().size(); i++) {
			final NEATNeuronGene neuronGene = (NEATNeuronGene) neuronsChromosome
					.getGene(i);
			if (neuronGene.getId() == neuronID) {
				return i;
			}
		}

		return -1;
	}

	/**
	 * @return The number of input neurons.
	 */
	public int getInputCount() {
		return inputCount;
	}

	/**
	 * @return THe links chromosome.
	 */
	public Chromosome getLinks() {
		return linksChromosome;
	}

	/**
	 * @return The network depth.
	 */
	public int getNetworkDepth() {
		return networkDepth;
	}

	/**
	 * @return The neurons chromosome.
	 */
	public Chromosome getNeurons() {
		return neuronsChromosome;
	}

	/**
	 * @return The number of genes in the links chromosome.
	 */
	public int getNumGenes() {
		return linksChromosome.size();
	}

	/**
	 * @return The output count.
	 */
	public int getOutputCount() {
		return outputCount;
	}

	/**
	 * @return The species ID.
	 */
	public long getSpeciesID() {
		return speciesID;
	}

	/**
	 * Get the specified split y.
	 * @param nd The neuron.
	 * @return The split y.
	 */
	public double getSplitY(final int nd) {
		return ((NEATNeuronGene) neuronsChromosome.get(nd)).getSplitY();
	}

	/**
	 * Determine if this is a duplicate link.
	 * @param fromNeuronID The from neuron id.
	 * @param toNeuronID The to neuron id.
	 * @return True if this is a duplicate link.
	 */
	public boolean isDuplicateLink(final long fromNeuronID, final long toNeuronID) {
		for (final Gene gene : getLinks().getGenes()) {
			final NEATLinkGene linkGene = (NEATLinkGene) gene;
			if ((linkGene.getFromNeuronID() == fromNeuronID)
					&& (linkGene.getToNeuronID() == toNeuronID)) {
				return true;
			}
		}

		return false;
	}

	/**
	 * Mutate the activation response.
	 * @param mutateRate The mutation rate.
	 * @param maxPertubation The maximum to perturb it by.
	 */
	public void mutateActivationResponse(final double mutateRate,
			final double maxPertubation) {
		for (final Gene gene : neuronsChromosome.getGenes()) {
			if (Math.random() < mutateRate) {
				final NEATNeuronGene neuronGene = (NEATNeuronGene) gene;
				neuronGene.setActivationResponse(neuronGene
						.getActivationResponse()
						+ RangeRandomizer.randomize(-1, 1) * maxPertubation);
			}
		}
	}

	/**
	 * Mutate the weights.
	 * @param mutateRate The mutation rate.
	 * @param probNewMutate The probability of a whole new weight.
	 * @param maxPertubation The max perturbation.
	 */
	public void mutateWeights(final double mutateRate, final double probNewMutate,
			final double maxPertubation) {
		for (final Gene gene : linksChromosome.getGenes()) {
			final NEATLinkGene linkGene = (NEATLinkGene) gene;
			if (Math.random() < mutateRate) {
				if (Math.random() < probNewMutate) {
					linkGene.setWeight(RangeRandomizer.randomize(-1, 1));
				} else {
					linkGene
							.setWeight(linkGene.getWeight()
									+ RangeRandomizer.randomize(-1, 1)
									* maxPertubation);
				}
			}
		}
	}

	/**
	 * @param networkDepth
	 *            the networkDepth to set
	 */
	public void setNetworkDepth(final int networkDepth) {
		this.networkDepth = networkDepth;
	}

	/**
	 * Set the species id.
	 * @param species The species id.
	 */
	public void setSpeciesID(final long species) {
		speciesID = species;
	}

	/**
	 * Sort the genes.
	 */
	public void sortGenes() {
		Collections.sort(linksChromosome.getGenes());
	}

	/**
	 * @return This genome as a string.
	 */
	@Override
	public String toString() {
		final StringBuilder result = new StringBuilder();
		result.append("[NEATGenome:");
		result.append(getGenomeID());
		result.append(",fitness=");
		result.append(getScore());
		result.append(")");
		return result.toString();
	}

}
