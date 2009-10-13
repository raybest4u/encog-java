package org.encog.normalize.output.nominal;

import java.util.ArrayList;
import java.util.List;

import org.encog.normalize.input.InputField;
import org.encog.normalize.output.OutputField;

public class OutputOneOf implements OutputField {

	private final List<NominalItem> items = new ArrayList<NominalItem>();

	public void addItem(final InputField inputField, final double low, final double high, double trueValue,double falseValue) {
		final NominalItem item = new NominalItem(inputField, low, high, trueValue, falseValue );
		this.items.add(item);
	}
	
	public void addItemBiPolar(InputField inputField, final double low, final double high)
	{
		addItem(inputField,low,high,1,-1);
	}
	
	public void addItemTF(InputField inputField, final double low, final double high)
	{
		addItem(inputField,low,high,1,0);
	}

	public double calculate(int subfield) {
		NominalItem item = this.items.get(subfield);
		return item.calculate();
	}
	
	public int getSubfieldCount()
	{
		return this.items.size();
	}

}
