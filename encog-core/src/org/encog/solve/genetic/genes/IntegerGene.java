package org.encog.solve.genetic.genes;


public class IntegerGene extends BasicGene {
	private int value;

	public int getValue() {
		return value;
	}

	public void setValue(int value) {
		this.value = value;
	}
	
	public void copy(Gene gene) {
		this.value = ((IntegerGene)gene).getValue();
		
	}
	
	public boolean equals(Object obj)
	{
		if( obj instanceof IntegerGene )
		{
			return(((IntegerGene)obj).getValue()==this.value);
		}
		else
			return false;
	}
	
	public String toString()
	{
		return ""+value;
	}
	
}
