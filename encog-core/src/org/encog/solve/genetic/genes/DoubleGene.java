package org.encog.solve.genetic.genes;


public class DoubleGene implements Gene {
	private double value;

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

	public void copy(Gene gene) {
		this.value = ((DoubleGene)gene).getValue();
		
	}
	
	public String toString()
	{
		return ""+value;
	}

	public int compareTo(Gene arg0) {
		return 0;
	}
	
}
