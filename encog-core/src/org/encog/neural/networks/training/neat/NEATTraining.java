package org.encog.neural.networks.training.neat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.encog.math.randomize.RangeRandomizer;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.CalculateScore;
import org.encog.neural.networks.training.Strategy;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.neat.NEATInnovationDB;

public class NEATTraining implements Train {

	private final int inputCount;
	private final int outputCount;
	private final List<NEATGenome> population = new ArrayList<NEATGenome>();
	private final List<NEATGenome> bestGenomes = new ArrayList<NEATGenome>();
	private final NEATInnovationDB innovations;
	private final List<SplitDepth> splits;
	private final List<NEATSpecies> species = new ArrayList<NEATSpecies>();
	private final CalculateScore calculateScore;
	private double bestEverFitness;
	private double totalFitAdjustment;
	private double averageFitAdjustment;

	private int currentGenomeID = 1;
	private int currentSpeciesID = 1;

	private int paramNumBestGenomes = 4;
	private double paramCompatibilityThreshold = 0.26;
	private int paramMaxNumberOfSpecies = 0;
	private int paramNumGensAllowedNoImprovement = 15;
	private double paramCrossoverRate = 0.7;
	private double paramMaxPermittedNeurons = 100;

	private double paramChanceAddNode = 0.04;
	private int paramNumTrysToFindOldLink = 5;

	private double paramChanceAddLink = 0.07;
	private double paramChanceAddRecurrentLink = 0.05;
	private int paramNumTrysToFindLoopedLink = 5;
	private int paramNumAddLinkAttempts = 5;

	private double paramMutationRate = 0.2;
	private double paramProbabilityWeightReplaced = 0.1;
	private double paramMaxWeightPerturbation = 0.5;

	private double paramActivationMutationRate = 0.1;
	private double paramMaxActivationPerturbation = 0.1;

	private int paramYoungBonusAgeThreshhold = 10;
	private double paramYoungFitnessBonus = 0.3;
	private int paramOldAgeThreshold = 50;
	private double paramOldAgePenalty = 0.3;
	private double paramSurvivalRate = 0.2;

	private final NEATGenomeComparator comparator;

	public NEATTraining(CalculateScore calculateScore, int inputCount,
			int outputCount, int populationSize) {
		this.calculateScore = calculateScore;
		this.inputCount = inputCount;
		this.outputCount = outputCount;

		this.comparator = new NEATGenomeComparator(calculateScore);

		// create the initial population
		for (int i = 0; i < populationSize; i++) {
			population.add(new NEATGenome(this, assignGenomeID(), inputCount,
					outputCount));
		}

		NEATGenome genome = new NEATGenome(this, 1, inputCount, outputCount);

		this.innovations = new NEATInnovationDB(genome.getLinks(), genome
				.getNeurons());

		this.splits = split(null, 0, 1, 0);

		if (this.calculateScore.shouldMinimize())
			this.bestEverFitness = Double.MAX_VALUE;
		else
			this.bestEverFitness = Double.MIN_VALUE;
	}

	public List<BasicNetwork> createNetworks() {
		List<BasicNetwork> result = new ArrayList<BasicNetwork>();

		for (NEATGenome genome : this.population) {
			calculateNetDepth(genome);
			BasicNetwork net = genome.createNetwork();

			result.add(net);
		}

		return result;
	}

	private void calculateNetDepth(NEATGenome genome) {
		int maxSoFar = 0;

		for (int nd = 0; nd < genome.getNeurons().size(); ++nd) {
			for (SplitDepth split : this.splits) {

				if ((genome.getSplitY(nd) == split.getValue())
						&& (split.getDepth() > maxSoFar)) {
					maxSoFar = split.getDepth();
				}
			}
		}

		genome.setNetworkDepth(maxSoFar + 2);
	}

	public void addNeuronID(int nodeID, List<Integer> vec) {
		for (int i = 0; i < vec.size(); i++) {
			if (vec.get(i) == nodeID) {
				return;
			}
		}

		vec.add(nodeID);

		return;
	}

	public int assignGenomeID() {
		return (this.currentGenomeID++);
	}

	public int getInputCount() {
		return inputCount;
	}

	public int getOutputCount() {
		return outputCount;
	}

	public void addStrategy(Strategy strategy) {
		// TODO Auto-generated method stub

	}

	public void finishTraining() {
		// TODO Auto-generated method stub

	}

	public double getError() {
		return this.population.get(0).getFitness();
	}

	public BasicNetwork getNetwork() {
		return this.population.get(0).createNetwork();
	}

	public List<Strategy> getStrategies() {
		// TODO Auto-generated method stub
		return null;
	}

	public void iteration() {

		for (NEATGenome genome : this.population) {
			BasicNetwork network = genome.createNetwork();
			double score = this.calculateScore.calculateScore(network);
			genome.setFitness(score);
		}

		resetAndKill();
		sortAndRecord();
		speciateAndCalculateSpawnLevels();

		List<NEATGenome> newPop = new ArrayList<NEATGenome>();

		int numSpawnedSoFar = 0;

		NEATGenome baby = null;

		for (NEATSpecies s : this.species) {
			if (numSpawnedSoFar < this.population.size()) {
				int numToSpawn = (int)Math.round(s.getNumToSpawn());

				boolean bChosenBestYet = false;

				while ((numToSpawn--) > 0) {
					if (!bChosenBestYet) {
						baby = s.getLeader();

						bChosenBestYet = true;
					}

					else {
						// if the number of individuals in this species is only
						// one
						// then we can only perform mutation
						if (s.getMembers().size() == 1) {
							// spawn a child
							baby = s.spawn();
						} else {
							NEATGenome g1 = s.spawn();

							if (Math.random() < this.paramCrossoverRate) {
								NEATGenome g2 = s.spawn();

								int NumAttempts = 5;

								while ((g1.getGenomeID() == g2.getGenomeID())
										&& ((NumAttempts--) > 0)) {
									g2 = s.spawn();
								}

								if (g1.getGenomeID() != g2.getGenomeID()) {
									baby = crossover(g1, g2);
								}
							}

							else {
								baby = g1;
							}
						}

						baby.setGenomeID(this.assignGenomeID());

						if (baby.getNeurons().size() < this.paramMaxPermittedNeurons) {
							baby.addNeuron(this.paramChanceAddNode,
									this.paramNumTrysToFindOldLink);
						}

						// now there's the chance a link may be added
						baby.addLink(this.paramChanceAddLink,
								this.paramChanceAddRecurrentLink,
								this.paramNumTrysToFindLoopedLink,
								this.paramNumAddLinkAttempts);

						// mutate the weights
						baby.mutateWeights(this.paramMutationRate,
								this.paramProbabilityWeightReplaced,
								this.paramMaxWeightPerturbation);

						baby.mutateActivationResponse(
								this.paramActivationMutationRate,
								this.paramMaxActivationPerturbation);
						
					}

					// sort the baby's genes by their innovation numbers
					baby.sortGenes();

					// add to new pop
					newPop.add(baby);

					++numSpawnedSoFar;

					if (numSpawnedSoFar == this.population.size()) {
						numToSpawn = 0;
					}
				}
			}
		}

		while (newPop.size() < this.population.size()) {
			newPop.add(tournamentSelection(this.population.size() / 5));
		}

		this.population.clear();
		this.population.addAll(newPop);
	}

	public void setError(double error) {
		// TODO Auto-generated method stub

	}

	private List<SplitDepth> split(List<SplitDepth> result, double low,
			double high, int depth) {
		if (result == null)
			result = new ArrayList<SplitDepth>();

		double span = high - low;

		result.add(new SplitDepth(low + span / 2, depth + 1));

		if (depth > 6) {
			return result;
		}

		else {
			split(result, low, low + span / 2, depth + 1);
			split(result, low + span / 2, high, depth + 1);
			return result;
		}
	}

	public NEATInnovationDB getInnovations() {
		return this.innovations;
	}

	public void sortAndRecord() {
		Collections.sort(this.population, this.comparator);

		this.bestEverFitness = this.comparator.bestScore(getError(),
				this.bestEverFitness);

		storeBestGenomes();
	}

	public void storeBestGenomes() {
		// clear old record
		this.bestGenomes.clear();

		for (int i = 0; i < this.paramNumBestGenomes; ++i) {
			this.bestGenomes.add(this.population.get(i));
		}
	}

	public List<BasicNetwork> getBestNetworksFromLastGeneration() {
		List<BasicNetwork> result = new ArrayList<BasicNetwork>();

		for (NEATGenome genome : this.population) {
			calculateNetDepth(genome);
			result.add(genome.createNetwork());
		}

		return result;
	}

	public void adjustSpeciesFitnesses() {
		for (NEATSpecies s : this.species) {
			s.adjustFitness();
		}
	}

	public void speciateAndCalculateSpawnLevels() {

		// calculate compatibility between genomes and species
		adjustCompatibilityThreshold();

		// assign genomes to species (if any exist)
		for (NEATGenome genome : this.population) {

			boolean added = false;

			for (NEATSpecies s : this.species) {
				double compatibility = genome.getCompatibilityScore(s
						.getLeader());

				if (compatibility <= this.paramCompatibilityThreshold) {
					s.addMember(genome);
					genome.setSpeciesID(s.getSpeciesID());
					added = true;
					break;
				}
			}

			// if this genome did not fall into any existing species, create a
			// new species
			if (!added) {
				this.species.add(new NEATSpecies(this, genome,
						assignSpeciesID()));
			}
		}

		adjustSpeciesFitnesses();

		for (NEATGenome genome : this.population) {
			this.totalFitAdjustment += genome.getAdjustedFitness();
		}

		this.averageFitAdjustment = this.totalFitAdjustment
				/ this.population.size();

		for (NEATGenome genome : this.population) {
			double toSpawn = genome.getAdjustedFitness()
					/ this.averageFitAdjustment;
			genome.setAmountToSpan(toSpawn);
		}

		for (NEATSpecies species : this.species) {
			species.calculateSpawnAmount();
		}
	}

	public void adjustCompatibilityThreshold() {

		if (this.paramMaxNumberOfSpecies < 1)
			return;

		double thresholdIncrement = 0.01;

		if (this.species.size() > this.paramMaxNumberOfSpecies) {
			this.paramCompatibilityThreshold += thresholdIncrement;
		}

		else if (this.species.size() < 2) {
			this.paramCompatibilityThreshold -= thresholdIncrement;
		}

	}

	public NEATGenome tournamentSelection(int numComparisons) {
		double bestFitnessSoFar = 0;

		int ChosenOne = 0;

		for (int i = 0; i < numComparisons; ++i) {
			int ThisTry = (int) RangeRandomizer.randomize(0, this.population
					.size() - 1);

			if (this.population.get(ThisTry).getFitness() > bestFitnessSoFar) {
				ChosenOne = ThisTry;

				bestFitnessSoFar = this.population.get(ThisTry).getFitness();
			}
		}

		return this.population.get(ChosenOne);
	}

	public NEATGenome crossover(NEATGenome mom, NEATGenome dad) {
		NEATParent best;

		if (mom.getFitness() == dad.getFitness()) {
			if (mom.getNumGenes() == dad.getNumGenes()) {
				if (Math.random() > 0)
					best = NEATParent.Mom;
				else
					best = NEATParent.Dad;
			}

			else {
				if (mom.getNumGenes() < dad.getNumGenes()) {
					best = NEATParent.Mom;
				}

				else {
					best = NEATParent.Dad;
				}
			}
		}

		else {
			if ( this.comparator.isBetterThan( mom.getFitness(), dad.getFitness() ) ) {
				best = NEATParent.Mom;
			}

			else {
				best = NEATParent.Dad;
			}
		}

		List<NEATNeuronGene> babyNeurons = new ArrayList<NEATNeuronGene>();
		List<NEATLinkGene> babyGenes = new ArrayList<NEATLinkGene>();

		List<Integer> vecNeurons = new ArrayList<Integer>();

		Iterator<NEATLinkGene> momIterator = mom.getLinks().iterator();
		Iterator<NEATLinkGene> dadIterator = dad.getLinks().iterator();

		NEATLinkGene selectedGene = null;

		while (momIterator.hasNext() || dadIterator.hasNext()) {
			NEATLinkGene momGene = null;
			NEATLinkGene dadGene = null;

			if (momIterator.hasNext())
				momGene = momIterator.next();

			if (dadIterator.hasNext())
				dadGene = dadIterator.next();

			if (momGene == null && dadGene != null) {
				if (best == NEATParent.Dad) {
					selectedGene = dadGene;
				}
			} else if (dadGene == null && momGene != null) {
				if (best == NEATParent.Mom) {
					selectedGene = momGene;
				}
			} else if (momGene.getInnovationID() < dadGene.getInnovationID()) {
				if (best == NEATParent.Mom) {
					selectedGene = momGene;
				}
			} else if (dadGene.getInnovationID() < momGene.getInnovationID()) {
				if (best == NEATParent.Dad) {
					selectedGene = dadGene;
				}
			} else if (dadGene.getInnovationID() == momGene.getInnovationID()) {
				if (Math.random() < 0.5f) {
					selectedGene = momGene;
				}

				else {
					selectedGene = dadGene;
				}

			}

			if (babyGenes.size() == 0) {
				babyGenes.add(selectedGene);
			}

			else {
				if (babyGenes.get(babyGenes.size() - 1).getInnovationID() != selectedGene
						.getInnovationID()) {
					babyGenes.add(selectedGene);
				}
			}

			// Check if we already have the nodes referred to in SelectedGene.
			// If not, they need to be added.
			addNeuronID(selectedGene.getFromNeuronID(), vecNeurons);
			addNeuronID(selectedGene.getToNeuronID(), vecNeurons);

		}// end while

		// now create the required nodes. First sort them into order
		Collections.sort(vecNeurons);

		for (int i = 0; i < vecNeurons.size(); i++) {
			babyNeurons.add(this.innovations.createNeuronFromID(vecNeurons
					.get(i)));
		}

		// finally, create the genome
		NEATGenome babyGenome = new NEATGenome(this, assignGenomeID(), babyNeurons,
				babyGenes, mom.getInputCount(), mom.getOutputCount());

		return babyGenome;
	}

	public void resetAndKill() {
		this.totalFitAdjustment = 0;
		this.averageFitAdjustment = 0;

		Object[] speciesArray = this.species.toArray();

		for (int i = 0; i < speciesArray.length; i++) {
			NEATSpecies s = (NEATSpecies) speciesArray[i];
			s.purge();

			if ((s.getGensNoImprovement() > this.paramNumGensAllowedNoImprovement)
					&& this.comparator.isBetterThan(this.bestEverFitness,s.getBestFitness()) ) {
				this.species.remove(s);
			}

		}

		for (NEATGenome genome : this.population) {
			genome.deleteNetwork();
		}
	}

	private int assignSpeciesID() {
		return this.currentSpeciesID++;
	}

	public NeuralDataSet getTraining() {
		// TODO Auto-generated method stub
		return null;
	}

	public int getParamNumBestGenomes() {
		return paramNumBestGenomes;
	}

	public void setParamNumBestGenomes(int paramNumBestGenomes) {
		this.paramNumBestGenomes = paramNumBestGenomes;
	}

	public double getParamCompatibilityThreshold() {
		return paramCompatibilityThreshold;
	}

	public void setParamCompatibilityThreshold(
			double paramCompatibilityThreshold) {
		this.paramCompatibilityThreshold = paramCompatibilityThreshold;
	}

	public int getParamMaxNumberOfSpecies() {
		return paramMaxNumberOfSpecies;
	}

	public void setParamMaxNumberOfSpecies(int paramMaxNumberOfSpecies) {
		this.paramMaxNumberOfSpecies = paramMaxNumberOfSpecies;
	}

	public int getParamNumGensAllowedNoImprovement() {
		return paramNumGensAllowedNoImprovement;
	}

	public void setParamNumGensAllowedNoImprovement(
			int paramNumGensAllowedNoImprovement) {
		this.paramNumGensAllowedNoImprovement = paramNumGensAllowedNoImprovement;
	}

	public double getParamCrossoverRate() {
		return paramCrossoverRate;
	}

	public void setParamCrossoverRate(double paramCrossoverRate) {
		this.paramCrossoverRate = paramCrossoverRate;
	}

	public double getParamMaxPermittedNeurons() {
		return paramMaxPermittedNeurons;
	}

	public void setParamMaxPermittedNeurons(double paramMaxPermittedNeurons) {
		this.paramMaxPermittedNeurons = paramMaxPermittedNeurons;
	}

	public double getParamChanceAddNode() {
		return paramChanceAddNode;
	}

	public void setParamChanceAddNode(double paramChanceAddNode) {
		this.paramChanceAddNode = paramChanceAddNode;
	}

	public int getParamNumTrysToFindOldLink() {
		return paramNumTrysToFindOldLink;
	}

	public void setParamNumTrysToFindOldLink(int paramNumTrysToFindOldLink) {
		this.paramNumTrysToFindOldLink = paramNumTrysToFindOldLink;
	}

	public double getParamChanceAddLink() {
		return paramChanceAddLink;
	}

	public void setParamChanceAddLink(double paramChanceAddLink) {
		this.paramChanceAddLink = paramChanceAddLink;
	}

	public double getParamChanceAddRecurrentLink() {
		return paramChanceAddRecurrentLink;
	}

	public void setParamChanceAddRecurrentLink(
			double paramChanceAddRecurrentLink) {
		this.paramChanceAddRecurrentLink = paramChanceAddRecurrentLink;
	}

	public int getParamNumTrysToFindLoopedLink() {
		return paramNumTrysToFindLoopedLink;
	}

	public void setParamNumTrysToFindLoopedLink(int paramNumTrysToFindLoopedLink) {
		this.paramNumTrysToFindLoopedLink = paramNumTrysToFindLoopedLink;
	}

	public int getParamNumAddLinkAttempts() {
		return paramNumAddLinkAttempts;
	}

	public void setParamNumAddLinkAttempts(int paramNumAddLinkAttempts) {
		this.paramNumAddLinkAttempts = paramNumAddLinkAttempts;
	}

	public double getParamMutationRate() {
		return paramMutationRate;
	}

	public void setParamMutationRate(double paramMutationRate) {
		this.paramMutationRate = paramMutationRate;
	}

	public double getParamProbabilityWeightReplaced() {
		return paramProbabilityWeightReplaced;
	}

	public void setParamProbabilityWeightReplaced(
			double paramProbabilityWeightReplaced) {
		this.paramProbabilityWeightReplaced = paramProbabilityWeightReplaced;
	}

	public double getParamMaxWeightPerturbation() {
		return paramMaxWeightPerturbation;
	}

	public void setParamMaxWeightPerturbation(double paramMaxWeightPerturbation) {
		this.paramMaxWeightPerturbation = paramMaxWeightPerturbation;
	}

	public double getParamActivationMutationRate() {
		return paramActivationMutationRate;
	}

	public void setParamActivationMutationRate(
			double paramActivationMutationRate) {
		this.paramActivationMutationRate = paramActivationMutationRate;
	}

	public double getParamMaxActivationPerturbation() {
		return paramMaxActivationPerturbation;
	}

	public void setParamMaxActivationPerturbation(
			double paramMaxActivationPerturbation) {
		this.paramMaxActivationPerturbation = paramMaxActivationPerturbation;
	}

	public int getParamYoungBonusAgeThreshhold() {
		return paramYoungBonusAgeThreshhold;
	}

	public void setParamYoungBonusAgeThreshhold(int paramYoungBonusAgeThreshhold) {
		this.paramYoungBonusAgeThreshhold = paramYoungBonusAgeThreshhold;
	}

	public double getParamYoungFitnessBonus() {
		return paramYoungFitnessBonus;
	}

	public void setParamYoungFitnessBonus(double paramYoungFitnessBonus) {
		this.paramYoungFitnessBonus = paramYoungFitnessBonus;
	}

	public int getParamOldAgeThreshold() {
		return paramOldAgeThreshold;
	}

	public void setParamOldAgeThreshold(int paramOldAgeThreshold) {
		this.paramOldAgeThreshold = paramOldAgeThreshold;
	}

	public double getParamOldAgePenalty() {
		return paramOldAgePenalty;
	}

	public void setParamOldAgePenalty(double paramOldAgePenalty) {
		this.paramOldAgePenalty = paramOldAgePenalty;
	}

	public double getParamSurvivalRate() {
		return paramSurvivalRate;
	}

	public void setParamSurvivalRate(double paramSurvivalRate) {
		this.paramSurvivalRate = paramSurvivalRate;
	}

	public NEATGenomeComparator getComparator() {
		return comparator;
	}

}
