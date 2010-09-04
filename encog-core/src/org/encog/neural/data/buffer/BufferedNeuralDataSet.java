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
package org.encog.neural.data.buffer;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.encog.engine.data.EngineData;
import org.encog.neural.data.Indexable;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataError;
import org.encog.neural.data.NeuralDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.persist.EncogCollection;
import org.encog.persist.EncogPersistedObject;
import org.encog.persist.Persistor;
import org.encog.persist.persistors.BufferedNeuralDataSetPersistor;

/**
 * This class is not memory based, so very long files can be used, without
 * running out of memory. This dataset uses a Encog binary training file as a
 * buffer.
 * 
 * When used with a slower access dataset, such as CSV, XML or SQL, where
 * parsing must occur, this dataset can be used to load from the slower dataset
 * and train at much higher speeds.
 * 
 * This class makes use of Java file channels for maximum file access
 * performance.
 * 
 * If you are going to create a binary file, by using the add methods, you must
 * call beginLoad to cause Encog to open an output file. Once the data has been
 * loaded, call endLoad. You can also use the BinaryDataLoader class, with a
 * CODEC, to load many other popular external formats.
 * 
 * The binary files produced by this class are in the Encog binary training
 * format, and can be used with any Encog platform. Encog binary files are
 * stored using "little endian" numbers.
 */
public class BufferedNeuralDataSet implements NeuralDataSet, Indexable,
		EncogPersistedObject {

	/**
	 * The version.
	 */
	private static final long serialVersionUID = 2577778772598513566L;

	/**
	 * Error message for ADD.
	 */
	public static final String ERROR_ADD = "Add can only be used after calling beginLoad.";

	/**
	 * Error message for REMOVE.
	 */
	public static final String ERROR_REMOVE = "Remove is not supported for BufferedNeuralDataSet.";

	private boolean loading;

	/**
	 * The file being used.
	 */
	private File file;

	private EncogEGBFile egb;

	/**
	 * Additional sets that were opened.
	 */
	private List<BufferedNeuralDataSet> additional = new ArrayList<BufferedNeuralDataSet>();

	/**
	 * The owner;
	 */
	private BufferedNeuralDataSet owner;

	/**
	 * The Encog persisted object name.
	 */
	private String name;

	/**
	 * The Encog persisted object description.
	 */
	private String description;

	/**
	 * The Encog persisted object collection.
	 */
	private EncogCollection collection;

	/**
	 * Construct the dataset using the specified binary file.
	 * 
	 * @param binaryFile
	 *            The file to use.
	 */
	public BufferedNeuralDataSet(File binaryFile) {
		this.file = binaryFile;
		this.egb = new EncogEGBFile(binaryFile);
		if( file.exists() )
			this.egb.open();
	}

	/**
	 * Open the binary file for reading.
	 */
	public void open() {
		this.egb.open();
	}

	/**
	 * @return An iterator.
	 */
	@Override
	public Iterator<NeuralDataPair> iterator() {
		return new BufferedDataSetIterator(this);
	}

	/**
	 * @return The record count.
	 */
	@Override
	public long getRecordCount() {
		if( this.egb==null)
			return 0;
		else
			return this.egb.getNumberOfRecords();
	}

	/**
	 * Read an individual record.
	 * 
	 * @param The
	 *            zero-based index. Specify 0 for the first record, 1 for the
	 *            second, and so on.
	 */
	@Override
	public void getRecord(long index, EngineData pair) {
		double[] inputTarget = pair.getInputArray();
		double[] idealTarget = pair.getIdealArray();

		this.egb.setLocation((int) index);
		this.egb.read(inputTarget);
		this.egb.read(idealTarget);
	}

	/**
	 * @return An additional training set.
	 */
	@Override
	public BufferedNeuralDataSet openAdditional() {

		BufferedNeuralDataSet result = new BufferedNeuralDataSet(this.file);
		result.setOwner(this);
		this.additional.add(result);
		return result;
	}

	/**
	 * Add only input data, for an unsupervised dataset.
	 * 
	 * @param data1
	 *            The data to be added.
	 */
	public void add(final NeuralData data1) {
		if (!this.loading) {
			throw new NeuralDataError(BufferedNeuralDataSet.ERROR_ADD);
		}

		egb.write(data1.getData());
	}

	/**
	 * Add both the input and ideal data.
	 * 
	 * @param inputData
	 *            The input data.
	 * @param idealData
	 *            The ideal data.
	 */
	public void add(final NeuralData inputData, final NeuralData idealData) {

		if (!this.loading) {
			throw new NeuralDataError(BufferedNeuralDataSet.ERROR_ADD);
		}

		this.egb.write(inputData.getData());
		this.egb.write(idealData.getData());
	}

	/**
	 * Add a data pair of both input and ideal data.
	 * 
	 * @param inputData
	 *            The pair to add.
	 */
	public void add(final NeuralDataPair pair) {
		if (!this.loading) {
			throw new NeuralDataError(BufferedNeuralDataSet.ERROR_ADD);
		}

		this.egb.write(pair.getInputArray());
		this.egb.write(pair.getIdealArray());

	}

	/**
	 * Close the dataset.
	 */
	@Override
	public void close() {

		Object[] obj = this.additional.toArray();

		for (int i = 0; i < obj.length; i++) {
			BufferedNeuralDataSet set = (BufferedNeuralDataSet) obj[i];
			set.close();
		}

		this.additional.clear();

		if (this.owner != null) {
			this.owner.removeAdditional(this);
		}

		this.egb.close();
		this.egb = null;
	}

	/**
	 * @return The ideal data size.
	 */
	@Override
	public int getIdealSize() {
		if( this.egb==null)
			return 0;
		else
			return this.egb.getIdealCount();
	}

	/**
	 * @return The input data size.
	 */
	@Override
	public int getInputSize() {
		if( this.egb==null)
			return 0;
		else
			return this.egb.getInputCount();
	}

	/**
	 * @return True if this dataset is supervised.
	 */
	@Override
	public boolean isSupervised() {
		if( this.egb==null)
			return false;
		else
		return this.egb.getIdealCount() > 0;
	}

	/**
	 * @return If this dataset was created by openAdditional, the set that
	 *         created this object is the owner. Return the owner.
	 */
	public BufferedNeuralDataSet getOwner() {
		return owner;
	}

	/**
	 * Set the owner of this dataset.
	 * 
	 * @param owner
	 *            The owner.
	 */
	public void setOwner(BufferedNeuralDataSet owner) {
		this.owner = owner;
	}

	/**
	 * Remove an additional dataset that was created.
	 * 
	 * @param child
	 *            The additional dataset to remove.
	 */
	public void removeAdditional(BufferedNeuralDataSet child) {
		synchronized (this) {
			this.additional.remove(child);
		}
	}

	/**
	 * Begin loading to the binary file. After calling this method the add
	 * methods may be called.
	 * 
	 * @param inputSize
	 *            The input size.
	 * @param idealSize
	 *            The ideal size.
	 */
	public void beginLoad(final int inputSize, final int idealSize) {
		this.egb.create(inputSize, idealSize);
		this.loading = true;
	}

	/**
	 * This method should be called once all the data has been loaded. The
	 * underlying file will be closed. The binary fill will then be opened for
	 * reading.
	 */
	public void endLoad() {
		if (!this.loading)
			throw new BufferedDataError("Must call beginLoad, before endLoad.");

		this.egb.close();

		open();

	}

	/**
	 * @return An Encog persistor for this object.
	 */
	@Override
	public Persistor createPersistor() {
		return new BufferedNeuralDataSetPersistor();
	}

	/**
	 * @return The name of this object.
	 */
	public String getName() {
		return name;
	}

	/**
	 * Set the name of this object, used for Encog persistance.
	 * 
	 * @param name
	 *            The name of this object.
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return The description of this object.
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * Set the description of this object.
	 * 
	 * @param The
	 *            description.
	 */
	public void setDescription(String description) {
		this.description = description;
	}

	/**
	 * @return The Encog persisted collection that this object belongs to.
	 */
	public EncogCollection getCollection() {
		return collection;
	}

	/**
	 * Set the Encog persisted collection that this object belongs to.
	 * 
	 * @param collection
	 *            The collection.
	 */
	public void setCollection(EncogCollection collection) {
		this.collection = collection;
	}

	/**
	 * @return The binary file used.
	 */
	public File getFile() {
		return this.file;
	}

	public EncogEGBFile getEGB() {
		return this.egb;
	}
}
