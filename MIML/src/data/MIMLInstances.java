/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package data;

import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import data.Bag;

/**
 * 
 * Class inheriting from MultiLabelnstances to represent MIML data.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class MIMLInstances extends MultiLabelInstances {

	/** For serialization */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor.
	 * 
	 * @param dataSet
	 *            A dataset of {@link Instances} with relational information.
	 * @param xmlLabelsDefFilePath
	 *            Path of .xml file with information about labels.
	 * @throws InvalidDataFormatException
	 *             To be handled in an upper level.
	 */
	public MIMLInstances(Instances dataSet, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(dataSet, xmlLabelsDefFilePath);
	}

	/**
	 * Constructor.
	 * 
	 * @param dataSet
	 *            A dataset of {@link Instances} with relational information.
	 * @param labelsMetaData
	 *            Information about labels.
	 * @throws InvalidDataFormatException
	 *             To be handled in an upper level.
	 */
	public MIMLInstances(Instances dataSet, LabelsMetaData labelsMetaData) throws InvalidDataFormatException {
		super(dataSet, labelsMetaData);
	}

	/**
	 * Constructor.
	 * 
	 * @param arffFilePath
	 *            Path of .arff file with Instances.
	 * @param xmlLabelsDefFilePath
	 *            Path of .xml file with information about labels.
	 * @throws InvalidDataFormatException
	 *             To be handled in an upper level.
	 */
	public MIMLInstances(String arffFilePath, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(arffFilePath, xmlLabelsDefFilePath);
	}

	/**
	 * Constructor.
	 * 
	 * @param arffFilePath
	 *            Path of .arff file with Instances.
	 * @param numLabelAttributes
	 *            Number of label attributes.
	 * @throws InvalidDataFormatException
	 *             To be handled in an upper level.
	 */
	public MIMLInstances(String arffFilePath, int numLabelAttributes) throws InvalidDataFormatException {
		super(arffFilePath, numLabelAttributes);
	}

	/**
	 * Gets a {@link Bag} (i.e. pattern) with a certain bagIndex.
	 * 
	 * @param bagIndex
	 *            Index of the bag.
	 * @return Bag If bagIndex exceeds the number of bags in the dataset. To be
	 *         handled in an upper level.
	 * @throws Exception
	 *             To be handled in an upper level.
	 */
	public Bag getBag(int bagIndex) throws Exception {
		if (bagIndex > this.getNumBags())
			throw new Exception("Out of bounds bagIndex: " + bagIndex + ". Actual numberOfBags: " + this.getNumBags());
		else {
			Instances aux = this.getDataSet();
			DenseInstance aux1 = (DenseInstance) aux.get(bagIndex);
			return new Bag(aux1);
		}
	}

	/**
	 * Gets a {@link Bag} with a certain bagIndex in the form of a set of
	 * {@link Instances} considering just the relational information. Neither
	 * identification attribute of the Bag nor label attributes are included.
	 * 
	 * @param bagIndex
	 *            Index of the bag
	 * @return a bag or an instance from the index of the dataset
	 * @throws Exception
	 *             If bagIndex exceeds the number of bags in the dataset. To be
	 *             handled in an upper level.
	 */
	public Instances getBagAsInstances(int bagIndex) throws Exception {
		if (bagIndex > this.getNumBags())
			throw new Exception("Out of bounds bagIndex: " + bagIndex + ". Actual numberOfBags: " + this.getNumBags());
		else {
			Instances bags = getBag(bagIndex).getBagAsInstances();
			return bags;
		}
	}

	/**
	 * Adds a Bag of Instances to the dataset.
	 *
	 * @param bag
	 *            A Bag of Instances.
	 */
	public void addBag(Bag bag) {
		this.getDataSet().add(bag);
	}

	/**
	 * Adds a Bag of Instances to the dataset in a certain index.
	 * 
	 * @param bag
	 *            A Bag of Instances.
	 * @param index
	 *            The index to insert the Bag.
	 */
	public void addInstance(Bag bag, int index) {
		this.getDataSet().add(index, bag);
	}

	/**
	 * Gets an instance of a bag.
	 * 
	 * @param bagIndex
	 *            The index of the bag in the data set,
	 * @param instanceIndex
	 *            Is the index of the instance in the bag.
	 * @return Instance
	 * @throws IndexOutOfBoundsException
	 *             To be handled in an upper level.
	 */
	public Instance getInstance(int bagIndex, int instanceIndex) throws IndexOutOfBoundsException {
		return this.getDataSet().instance(bagIndex).relationalValue(1).instance(instanceIndex);
	}

	/**
	 * Gets the number of bags of the dataset.
	 * 
	 * @return int
	 */
	public int getNumBags() {
		return this.getNumInstances();
	}

	/**
	 * Gets the number of instances of a bag.
	 * 
	 * @param bagIndex
	 *            A bag index.
	 * @return int
	 * @throws Exception
	 *             To be handled in an upper level.
	 */
	public int getNumInstances(int bagIndex) throws Exception {
		return this.getBag(bagIndex).relationalValue(1).numInstances();
	}

	/**
	 * Gets the number of attributes of the dataset considering label attributes
	 * and the relational attribute with bags as a single attribute. For
	 * instance, in relation above, the returned value is 6.
	 * 
	 * &#064;relation toy<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute bag relational<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * &#064;end bag<br>
	 * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * @return int
	 */
	public int getNumAttributes() {
		return this.getDataSet().numAttributes();
	}

	/**
	 * Gets the total number of attributes of the dataset. This number includes
	 * attributes corresponding to labels. Instead the relational attribute, the
	 * number of attributes contained in the relational attribute is considered.
	 * For instance, in the relation above, the output of the method is 8.<br>
	 * 
	 * &#064;relation toy<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute bag relational<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * &#064;end bag<br>
	 * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * 
	 * @return int
	 */
	public int getNumAttributesWithRelational() {
		return this.getDataSet().numAttributes() + this.getNumAttributesInABag() - 1;
	}

	/**
	 * Gets the number of attributes per bag. In MIML all bags have the same
	 * number of attributes.* For instance, in the relation above, the output of
	 * the method is 3.<br>
	 * 
	 * &#064;relation toy<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute bag relational<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * &#064;end bag<br>
	 * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * @return int
	 */
	public int getNumAttributesInABag() {
		return this.getDataSet().instance(0).relationalValue(1).numAttributes();
	}

	/**
	 * Returns the dataset as MultiLabelInstances.
	 * 
	 * @return MultiLabelInstances
	 */
	public MultiLabelInstances getMLDataSet() {
		return (MultiLabelInstances) this;
	}

}
