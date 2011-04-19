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
package org.encog.ml.data.basic;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.encog.EncogError;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.data.MLDataArray;
import org.encog.neural.data.NeuralDataSet;
import org.encog.util.EngineArray;
import org.encog.util.obj.ObjectCloner;

/**
 * neural data in an ArrayList. This class is memory based, so large enough
 * datasets could cause memory issues. Many other dataset types extend this
 * class.
 * 
 * @author jheaton
 */
public class BasicNeuralDataSet implements Serializable,
		NeuralDataSet {

	/**
	 * An iterator to be used with the BasicNeuralDataSet. This iterator does
	 * not support removes.
	 * 
	 * @author jheaton
	 */
	public class BasicNeuralIterator implements Iterator<MLDataPair> {

		/**
		 * The index that the iterator is currently at.
		 */
		private int currentIndex = 0;

		/**
		 * Is there more data for the iterator to read?
		 * 
		 * @return Returns true if there is more data to read.
		 */
		public boolean hasNext() {
			return this.currentIndex < BasicNeuralDataSet.this.data.size();
		}

		/**
		 * Read the next item.
		 * 
		 * @return The next item.
		 */
		public MLDataPair next() {
			if (!hasNext()) {
				return null;
			}

			return BasicNeuralDataSet.this.data.get(this.currentIndex++);
		}

		/**
		 * Removes are not supported.
		 */
		public void remove() {
			throw new EncogError("Called remove, unsupported operation.");
		}
	}

	/**
	 * The serial id.
	 */
	private static final long serialVersionUID = -2279722928570071183L;


	/**
	 * The data held by this object.
	 */
	private List<MLDataPair> data = new ArrayList<MLDataPair>();

	/**
	 * Default constructor.
	 */
	public BasicNeuralDataSet() {
	}

	/**
	 * Construct a data set from an input and idea array.
	 * 
	 * @param input
	 *            The input into the neural network for training.
	 * @param ideal
	 *            The ideal output for training.
	 */
	public BasicNeuralDataSet(final double[][] input, final double[][] ideal) {
		if (ideal != null) {
			for (int i = 0; i < input.length; i++) {
				final BasicMLDataArray inputData = new BasicMLDataArray(input[i]);
				final BasicMLDataArray idealData = new BasicMLDataArray(ideal[i]);
				this.add(inputData, idealData);
			}
		} else {
			for (final double[] element : input) {
				final BasicMLDataArray inputData = new BasicMLDataArray(element);
				this.add(inputData);
			}
		}
	}

	/**
	 * Construct a data set from an already created list. Mostly used to
	 * duplicate this class.
	 * 
	 * @param data
	 *            The data to use.
	 */
	public BasicNeuralDataSet(final List<MLDataPair> data) {
		this.data = data;
	}

	/**
	 * Copy whatever dataset type is specified into a memory dataset.
	 * @param set The dataset to copy.
	 */
	public BasicNeuralDataSet(NeuralDataSet set) {
		int inputCount = set.getInputSize();
		int idealCount = set.getIdealSize();
		
		for(MLDataPair pair: set) {
			
			BasicMLDataArray input = null;
			BasicMLDataArray ideal = null;
			
			if( inputCount>0 ) {
				input = new BasicMLDataArray(inputCount);
				EngineArray.arrayCopy(pair.getInputArray(), input.getData());
			}
			
			if( idealCount>0 ) {
				ideal = new BasicMLDataArray(idealCount);
				EngineArray.arrayCopy(pair.getIdealArray(), ideal.getData());
			}
			
			add(new BasicMLDataPair(input,ideal));
		}
	}

	/**
	 * Add input to the training set with no expected output. This is used for
	 * unsupervised training.
	 * 
	 * @param data
	 *            The input to be added to the training set.
	 */
	public void add(final MLDataArray data) {
		this.data.add(new BasicMLDataPair(data));
	}

	/**
	 * Add input and expected output. This is used for supervised training.
	 * 
	 * @param inputData
	 *            The input data to train on.
	 * @param idealData
	 *            The ideal data to use for training.
	 */
	public void add(final MLDataArray inputData, final MLDataArray idealData) {

		final MLDataPair pair = new BasicMLDataPair(inputData,
				idealData);
		this.data.add(pair);
	}

	/**
	 * Add a neural data pair to the list.
	 * 
	 * @param inputData
	 *            A NeuralDataPair object that contains both input and ideal
	 *            data.
	 */
	public void add(final MLDataPair inputData) {
		this.data.add(inputData);
	}

	/**
	 * @return A cloned copy of this object.
	 */
	@Override
	public Object clone() {
		return ObjectCloner.deepCopy(this);
	}

	/**
	 * Close this data set.
	 */
	public void close() {
		// nothing to close
	}

	/**
	 * Get the data held by this container.
	 * 
	 * @return the data
	 */
	public List<MLDataPair> getData() {
		return this.data;
	}

	/**
	 * Get the size of the ideal dataset. This is obtained from the first item
	 * in the list.
	 * 
	 * @return The size of the ideal data.
	 */
	public int getIdealSize() {
		if (this.data.isEmpty()) {
			return 0;
		}
		final MLDataPair first = this.data.get(0);
		if (first.getIdeal() == null) {
			return 0;
		}

		return first.getIdeal().size();
	}

	/**
	 * Get the size of the input dataset. This is obtained from the first item
	 * in the list.
	 * 
	 * @return The size of the input data.
	 */
	public int getInputSize() {
		if (this.data.isEmpty()) {
			return 0;
		}
		final MLDataPair first = this.data.get(0);
		return first.getInput().size();
	}


	/**
	 * Get a record by index into the specified pair.
	 * 
	 * @param index
	 *            The index to read.
	 * @param pair
	 *            The pair to hold the data.
	 */
	public void getRecord(final long index, final MLDataPair pair) {

		final MLDataPair source = this.data.get((int) index);
		pair.setInputArray(source.getInputArray());
		if (pair.getIdealArray() != null) {
			pair.setIdealArray(source.getIdealArray());
		}

	}

	/**
	 * @return The total number of records in the file.
	 */
	public long getRecordCount() {
		return this.data.size();
	}

	/**
	 * Determine if this neural data set is supervied. All of the pairs should
	 * be either supervised or not, so simply check the first pair. If the list
	 * is empty then assume unsupervised.
	 * 
	 * @return True if supervised.
	 */
	public boolean isSupervised() {
		if (this.data.size() == 0) {
			return false;
		}
		return this.data.get(0).isSupervised();
	}

	/**
	 * Create an iterator for this collection.
	 * 
	 * @return An iterator to access this collection.
	 */
	public Iterator<MLDataPair> iterator() {
		final BasicNeuralIterator result = new BasicNeuralIterator();
		return result;
	}

	/**
	 * Create an additional data set. It will use the same list.
	 * 
	 * @return The additional data set.
	 */
	public MLDataSet openAdditional() {
		return new BasicNeuralDataSet(this.data);
	}

	/**
	 * @param data
	 *            the data to set
	 */
	public void setData(final List<MLDataPair> data) {
		this.data = data;
	}

}
