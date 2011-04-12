/*
 * Encog(tm) Core v3.0 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
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
package org.encog.app.analyst.script.normalize;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.encog.Encog;
import org.encog.app.analyst.AnalystError;
import org.encog.app.analyst.util.CSVHeaders;
import org.encog.app.csv.basic.BasicFile;
import org.encog.app.csv.normalize.ClassItem;
import org.encog.app.csv.normalize.NormalizationAction;
import org.encog.app.quant.QuantError;
import org.encog.engine.util.EngineArray;
import org.encog.mathutil.Equilateral;
import org.encog.util.csv.CSVFormat;

/**
 * Holds a field to be analyzed.
 *
 */
public class AnalystField {
	/**
	 * The actual high from the sample data.
	 */
	private double actualHigh;

	/**
	 * The actual low from the sample data.
	 */
	private double actualLow;

	/**
	 * The desired normalized high.
	 */
	private double normalizedHigh;

	/**
	 * The desired normalized low from the sample data.
	 */
	private double normalizedLow;

	/**
	 * The action that should be taken on this column.
	 */
	private NormalizationAction action;

	/**
	 * The name of this column.
	 */
	private String name;

	/**
	 * The list of classes.
	 */
	private final List<ClassItem> classes = new ArrayList<ClassItem>();

	/**
	 * If equilateral classification is used, this is the Equilateral object.
	 */
	private Equilateral eq;

	/**
	 * Allows the index of a field to be looked up.
	 */
	private final Map<String, Integer> lookup = new HashMap<String, Integer>();

	/**
	 * True, if this is an output field.
	 */
	private boolean output;
	
	/**
	 * The time slice number.
	 */
	private int timeSlice;

	/**
	 * Construct the object with a range of 1 and -1.
	 */
	public AnalystField() {
		this(1, -1);
	}

	public AnalystField(final AnalystField field) {
		this.actualHigh = field.actualHigh;
		this.actualLow = field.actualLow;
		this.normalizedHigh = field.normalizedHigh;
		this.normalizedLow = field.normalizedLow;
		this.action = field.action;
		this.name = field.name;
		this.output = field.output;
		this.timeSlice = field.timeSlice;
	}

	/**
	 * Construct the object.
	 * 
	 * @param normalizedHigh
	 *            The normalized high.
	 * @param normalizedLow
	 *            The normalized low.
	 */
	public AnalystField(final double normalizedHigh, final double normalizedLow) {
		this.normalizedHigh = normalizedHigh;
		this.normalizedLow = normalizedLow;
		this.actualHigh = Double.MIN_VALUE;
		this.actualLow = Double.MAX_VALUE;
		this.action = NormalizationAction.Normalize;
	}

	/**
	 * Construct an object.
	 * 
	 * @param action
	 *            The desired action.
	 * @param name
	 *            The name of this column.
	 */
	public AnalystField(final NormalizationAction action, final String name) {
		this(action, name, 0, 0, 0, 0);
	}

	/**
	 * Construct the field, with no defaults.
	 * 
	 * @param action
	 *            The normalization action to take.
	 * @param name
	 *            The name of this field.
	 * @param ahigh
	 *            The actual high.
	 * @param alow
	 *            The actual low.
	 * @param nhigh
	 *            The normalized high.
	 * @param nlow
	 *            The normalized low.
	 */
	public AnalystField(final NormalizationAction action, final String name,
			final double ahigh, final double alow, final double nhigh,
			final double nlow) {
		this.action = action;
		this.actualHigh = ahigh;
		this.actualLow = alow;
		this.normalizedHigh = nhigh;
		this.normalizedLow = nlow;
		this.name = name;
	}

	public AnalystField(final String name, final NormalizationAction action,
			final double high, final double low) {
		this.name = name;
		this.action = action;
		this.normalizedHigh = high;
		this.normalizedLow = low;
	}

	public void addRawHeadings(final StringBuilder line, final String prefix,
			final CSVFormat format) {
		final int subFields = getColumnsNeeded();

		for (int i = 0; i < subFields; i++) {
			final String str = CSVHeaders.tagColumn(this.name, i,
					this.timeSlice, subFields > 1);
			BasicFile.appendSeparator(line, format);
			line.append('\"');
			if (prefix != null) {
				line.append(prefix);
			}
			line.append(str);
			line.append('\"');
		}
	}

	/**
	 * Analyze the specified value. Adjust min/max as needed. Usually used only
	 * internally.
	 * 
	 * @param d
	 *            The value to analyze.
	 */
	public void analyze(final double d) {
		this.actualHigh = Math.max(this.actualHigh, d);
		this.actualLow = Math.min(this.actualLow, d);
	}

	/**
	 * Denormalize the specified value.
	 * 
	 * @param value
	 *            The value to normalize.
	 * @return The normalized value.
	 */
	public double deNormalize(final double value) {
		final double result = ((this.actualLow - this.actualHigh) * value
				- this.normalizedHigh * this.actualLow + this.actualHigh
				* this.normalizedLow)
				/ (this.normalizedLow - this.normalizedHigh);
		return result;
	}

	/**
	 * Determine what class the specified data belongs to.
	 * 
	 * @param data
	 *            The data to analyze.
	 * @return The class the data belongs to.
	 */
	public ClassItem determineClass(final double[] data) {
		int resultIndex = 0;

		switch (this.action) {
		case Equilateral:
			resultIndex = this.eq.decode(data);
			break;
		case OneOf:
			resultIndex = EngineArray.indexOfLargest(data);
			break;
		case SingleField:
			resultIndex = (int) data[0];
			break;
		}

		return this.classes.get(resultIndex);
	}

	public ClassItem determineClass(final int pos, final double[] data) {
		int resultIndex = 0;
		final double[] d = new double[getColumnsNeeded()];
		EngineArray.arrayCopy(data, pos, d, 0, d.length);

		switch (this.action) {
		case Equilateral:
			resultIndex = this.eq.decode(d);
			break;
		case OneOf:
			resultIndex = EngineArray.indexOfLargest(d);
			break;
		case SingleField:
			resultIndex = (int) d[0];
			break;
		default:
			throw new AnalystError("Invalid action: " + this.action);
		}

		if (resultIndex < 0) {
			return null;
		}

		return this.classes.get(resultIndex);
	}

	/**
	 * Encode the class.
	 * 
	 * @param classNumber
	 *            The class number.
	 * @return The encoded class.
	 */
	public double[] encode(final int classNumber) {
		switch (this.action) {
		case OneOf:
			return encodeOneOf(classNumber);
		case Equilateral:
			return encodeEquilateral(classNumber);
		case SingleField:
			return encodeSingleField(classNumber);
		default:
			return null;
		}
	}

	public double[] encode(final String str) {
		int classNumber = lookup(str);
		if (classNumber == -1) {
			try {
				classNumber = Integer.parseInt(str);
			} catch (final NumberFormatException ex) {
				throw new QuantError("Can't determine class for: " + str);
			}
		}
		return encode(classNumber);

	}

	/**
	 * Perform an equilateral encode.
	 * 
	 * @param classNumber
	 *            The class number.
	 * @return The class to encode.
	 */
	public double[] encodeEquilateral(final int classNumber) {
		return this.eq.encode(classNumber);
	}

	/**
	 * Perform the encoding for "one of".
	 * 
	 * @param classNumber
	 *            The class number.
	 * @return The encoded columns.
	 */
	public double[] encodeOneOf(final int classNumber) {
		final double[] result = new double[getColumnsNeeded()];

		for (int i = 0; i < this.classes.size(); i++) {
			if (i == classNumber) {
				result[i] = this.normalizedHigh;
			} else {
				result[i] = this.normalizedLow;
			}
		}
		return result;
	}

	/**
	 * Encode a single field.
	 * 
	 * @param classNumber
	 *            The class number to encode.
	 * @return The encoded columns.
	 */
	public double[] encodeSingleField(final int classNumber) {
		final double[] d = new double[1];
		d[0] = classNumber;
		return d;
	}

	/**
	 * Fix normalized fields that have a single value for the min/max. Separate
	 * them by 2 units.
	 */
	public void fixSingleValue() {
		if (this.action == NormalizationAction.Normalize) {
			if (Math.abs(this.actualHigh - this.actualLow) < Encog.DEFAULT_DOUBLE_EQUAL) {
				this.actualHigh += 1;
				this.actualLow -= 1;
			}
		}
	}

	/**
	 * @return The action for the field.
	 */
	public NormalizationAction getAction() {
		return this.action;
	}

	/**
	 * @return The actual high for the field.
	 */
	public double getActualHigh() {
		return this.actualHigh;
	}

	/**
	 * @return The actual low for the field.
	 */
	public double getActualLow() {
		return this.actualLow;
	}

	public List<ClassItem> getClasses() {
		return this.classes;
	}

	/**
	 * @return Returns the number of columns needed for this classification. The
	 *         number of columns needed will vary, depending on the
	 *         classification method used.
	 */
	public int getColumnsNeeded() {
		switch (this.action) {
		case Ignore:
			return 0;
		case Equilateral:
			return this.classes.size() - 1;
		case OneOf:
			return this.classes.size();
		default:
			return 1;
		}

	}

	public Equilateral getEq() {
		return this.eq;
	}

	/**
	 * @return The name of the field.
	 */
	public String getName() {
		return this.name;
	}

	/**
	 * @return The normalized high for the field.
	 */
	public double getNormalizedHigh() {
		return this.normalizedHigh;
	}

	/**
	 * @return The normalized low for the neural network.
	 */
	public double getNormalizedLow() {
		return this.normalizedLow;
	}

	/**
	 * @return the timeSlice
	 */
	public int getTimeSlice() {
		return this.timeSlice;
	}

	/**
	 * Init any internal structures.
	 * 
	 * @param owner
	 */
	public void init() {

		if (this.action == NormalizationAction.Equilateral) {
			if (this.classes.size() < 3) {
				throw new QuantError(
						"There must be at least three classes to make use of equilateral normalization.");
			}

			this.eq = new Equilateral(this.classes.size(), this.normalizedHigh,
					this.normalizedLow);
		}

		// build lookup map
		for (int i = 0; i < this.classes.size(); i++) {
			this.lookup.put(this.classes.get(i).getName(), this.classes.get(i)
					.getIndex());
		}
	}

	public boolean isClassify() {
		return (this.action == NormalizationAction.Equilateral)
				|| (this.action == NormalizationAction.OneOf)
				|| (this.action == NormalizationAction.SingleField);
	}

	public boolean isIgnored() {
		return this.action == NormalizationAction.Ignore;
	}

	public boolean isInput() {
		return !this.output;
	}

	public boolean isOutput() {
		return this.output;
	}

	/**
	 * Lookup the specified field.
	 * 
	 * @param str
	 *            The name of the field to lookup.
	 * @return The index of the field, or -1 if not found.
	 */
	public int lookup(final String str) {
		if (!this.lookup.containsKey(str)) {
			return -1;
		}
		return this.lookup.get(str);
	}

	public void makeClass(final NormalizationAction action,
			final int classFrom, final int classTo, final int high,
			final int low) {

		if ((action != NormalizationAction.Equilateral)
				&& (action != NormalizationAction.OneOf)
				&& (action != NormalizationAction.SingleField)) {
			throw new QuantError("Unsupported normalization type");
		}

		this.action = action;
		this.classes.clear();
		this.normalizedHigh = high;
		this.normalizedLow = low;
		this.actualHigh = 0;
		this.actualLow = 0;

		int index = 0;
		for (int i = classFrom; i < classTo; i++) {
			this.classes.add(new ClassItem("" + i, index++));
		}

	}

	public void makeClass(final NormalizationAction action, final String[] cls,
			final double high, final double low) {
		if ((action != NormalizationAction.Equilateral)
				&& (action != NormalizationAction.OneOf)
				&& (action != NormalizationAction.SingleField)) {
			throw new QuantError("Unsupported normalization type");
		}

		this.action = action;
		this.classes.clear();
		this.normalizedHigh = high;
		this.normalizedLow = low;
		this.actualHigh = 0;
		this.actualLow = 0;

		for (int i = 0; i < cls.length; i++) {
			this.classes.add(new ClassItem(cls[i], i));
		}

	}

	/**
	 * Make this a pass-through field.
	 */
	public void makePassThrough() {
		this.normalizedHigh = 0;
		this.normalizedLow = 0;
		this.actualHigh = 0;
		this.actualLow = 0;
		this.action = NormalizationAction.PassThrough;
	}

	/**
	 * Normalize the specified value.
	 * 
	 * @param value
	 *            The value to normalize.
	 * @return The normalized value.
	 */
	public double normalize(final double value) {
		return ((value - this.actualLow) / (this.actualHigh - this.actualLow))
				* (this.normalizedHigh - this.normalizedLow)
				+ this.normalizedLow;
	}

	/**
	 * Set the action for the field.
	 * 
	 * @param action
	 *            The action for the field.
	 */
	public void setAction(final NormalizationAction action) {
		this.action = action;
	}

	/**
	 * Set the actual high for the field.
	 * 
	 * @param actualHigh
	 *            The actual high for the field.
	 */
	public void setActualHigh(final double actualHigh) {
		this.actualHigh = actualHigh;
	}

	/**
	 * Set the actual low for the field.
	 * 
	 * @param actualLow
	 *            The actual low for the field.
	 */
	public void setActualLow(final double actualLow) {
		this.actualLow = actualLow;
	}

	/**
	 * Set the name of the field.
	 * 
	 * @param name
	 *            The name of the field.
	 */
	public void setName(final String name) {
		this.name = name;
	}

	/**
	 * Set the normalized high for the field.
	 * 
	 * @param normalizedHigh
	 *            The normalized high for the field.
	 */
	public void setNormalizedHigh(final double normalizedHigh) {
		this.normalizedHigh = normalizedHigh;
	}

	/**
	 * Set the normalized low for the field.
	 * 
	 * @param normalizedLow
	 *            The normalized low for the field.
	 */
	public final void setNormalizedLow(final double normalizedLow) {
		this.normalizedLow = normalizedLow;
	}

	/**
	 * Set if this is an output field.
	 * @param b True, if this is output.
	 */
	public final void setOutput(final boolean b) {
		this.output = b;
	}

	/**
	 * @param theTimeSlice
	 *            the timeSlice to set
	 */
	public final void setTimeSlice(final int theTimeSlice) {
		this.timeSlice = theTimeSlice;
	}

	/** {@inheritDoc} */
	@Override
	public final String toString() {
		final StringBuilder result = new StringBuilder("[");
		result.append(getClass().getSimpleName());
		result.append(" name=");
		result.append(this.name);
		result.append(", actualHigh=");
		result.append(this.actualHigh);
		result.append(", actualLow=");
		result.append(this.actualLow);

		result.append("]");
		return result.toString();
	}
}
