package org.encog.solve.genetic.genome;

import java.util.ArrayList;
import java.util.List;

import org.encog.solve.genetic.GeneticAlgorithm;
import org.encog.solve.genetic.GeneticError;

public abstract class BasicGenome implements Genome {
	
	private final List<Chromosome> chromosomes = new ArrayList<Chromosome>();
	private double score;
	private final GeneticAlgorithm geneticAlgorithm;
	private Object organism;
	private long genomeID;
	private double adjustedScore;
	private double amountToSpawn;
	
	public BasicGenome(GeneticAlgorithm geneticAlgorithm)
	{
		this.geneticAlgorithm = geneticAlgorithm;
	}
	
	public List<Chromosome> getChromosomes()
	{
		return this.chromosomes;
	}
	
	/**
	 * Used to compare two chromosomes. Used to sort by score.
	 * 
	 * @param other
	 *            The other chromosome to compare.
	 * @return The value 0 if the argument is a chromosome that has an equal
	 *         score to this chromosome; a value less than 0 if the argument is
	 *         a chromosome with a score greater than this chromosome; and a
	 *         value greater than 0 if the argument is a chromosome what a score
	 *         less than this chromosome.
	 */
	public int compareTo(final Genome other) {

		if (this.geneticAlgorithm.getCalculateScore().shouldMinimize()) {
			if (getScore() > other.getScore()) {
				return 1;
			}
			return -1;
		} else {
			if (getScore() > other.getScore()) {
				return -1;
			}
			return 1;

		}
	}

	public int calculateGeneCount() {
		double result = 0;
		
		for( Chromosome chromosome: this.chromosomes )
		{
			result+=chromosome.getGenes().size();
		}
		return (int)(result/this.getChromosomes().size());
	}

	public double getScore() {
		return this.score;
	}

	public void mate(Genome father, Genome child1, Genome child2) {
		int motherChromosomes = this.getChromosomes().size();
		int fatherChromosomes = father.getChromosomes().size();
		
		if( motherChromosomes!=fatherChromosomes )
		{
			throw new GeneticError("Mother and father must have same chromosome count, Mother:" + motherChromosomes + ",Father:" + fatherChromosomes);
		}
		
		for(int i=0;i<fatherChromosomes;i++)
		{
			Chromosome motherChromosome = this.chromosomes.get(i);
			Chromosome fatherChromosome = father.getChromosomes().get(i);
			Chromosome offspring1Chromosome = child1.getChromosomes().get(i);
			Chromosome offspring2Chromosome = child2.getChromosomes().get(i);
			
			this.geneticAlgorithm.getCrossover().mate(
				motherChromosome, 
				fatherChromosome, 
				offspring1Chromosome, 
				offspring2Chromosome);
			
			if( Math.random()<this.geneticAlgorithm.getMutationPercent() )
			{
				this.geneticAlgorithm.getMutate().performMutation(offspring1Chromosome);
			}
			
			if( Math.random()<this.geneticAlgorithm.getMutationPercent() )
			{
				this.geneticAlgorithm.getMutate().performMutation(offspring2Chromosome);
			}
		}
		
		child1.decode();
		child2.decode();
		this.geneticAlgorithm.calculateScore(child1);
		this.geneticAlgorithm.calculateScore(child2);
	}
	
	public void setScore(double score)
	{
		this.score = score;
	}
	
	/**
	 * Convert the chromosome to a string.
	 * 
	 * @return The chromosome as a string.
	 */
	@Override
	public String toString() {
		final StringBuilder builder = new StringBuilder();
		builder.append("[BasicGenome: score=");
		builder.append(getScore());
		return builder.toString();
	}

	public Object getOrganism() {
		return organism;
	}

	public void setOrganism(Object organism) {
		this.organism = organism;
	}

	public GeneticAlgorithm getGeneticAlgorithm() {
		return geneticAlgorithm;
	}

	public long getGenomeID() {
		return genomeID;
	}

	public void setGenomeID(long genomeID) {
		this.genomeID = genomeID;
	}

	public double getAdjustedScore() {
		return adjustedScore;
	}

	public void setAdjustedScore(double adjustedScore) {
		this.adjustedScore = adjustedScore;
	}

	public double getAmountToSpawn() {
		return amountToSpawn;
	}

	public void setAmountToSpawn(double amountToSpawn) {
		this.amountToSpawn = amountToSpawn;
	}

	
	

}
