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
 * Class inheriting from the MultiLabelnstances to represent multi-instance
 * multi-label problems.
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
	 *            The set of bags of instances
	 * @param xmlLabelsDefFilePath
	 *            Path of .xml file with information about labels
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(Instances dataSet, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(dataSet, xmlLabelsDefFilePath);
	}

	/**
	 * Constructor.
	 * 
	 * @param dataSet
	 *            The set of bags of instances
	 * @param labelsMetaData
	 *            Information about labels
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(Instances dataSet, LabelsMetaData labelsMetaData) throws InvalidDataFormatException {
		super(dataSet, labelsMetaData);
	}

	/**
	 * Constructor.
	 * 
	 * @param arffFilePath
	 *            Path of .arff file with instances
	 * @param xmlLabelsDefFilePath
	 *            Path of .xml file with information about labels
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(String arffFilePath, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(arffFilePath, xmlLabelsDefFilePath);
	}

	/**
	 * Constructor.
	 * 
	 * @param arffFilePath
	 *            Path of .arff file with instances
	 * @param numLabelAttributes
	 *            Number of Attributes
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(String arffFilePath, int numLabelAttributes) throws InvalidDataFormatException {
		super(arffFilePath, numLabelAttributes);
	}

	/**
	 * Gets a bag or pattern from MultiLabel instances from a index
	 * 
	 * @param bagIndex
	 *            Index of bag
	 * @return a bag or an instance from the index of the dataset
	 * @throws Exception
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
	 * Gets a bag from a index in the form of a set of instances considering just the relational information.
	 * Identification of bag and information about labels is not included.
	 * 
	 * @param bagIndex
	 *            Index of bag
	 * @return a bag or an instance from the index of the dataset
	 * @throws Exception Potential exception thrown. To be handled in an upper level.
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
	 * Adds a bag of instances to the dataset.
	 *
	 * @param bag
	 *            A bag of instances.
	 */
	public void addBag(Bag bag) {
		this.getDataSet().add(bag);
	}

	/**
	 * Adds a bag of instances to the dataset in a certain index.
	 * 
	 * @param bag
	 *            A bag of instanes
	 * @param index
	 *            The index to insert the new bag
	 */
	public void addInstance(Bag bag, int index) {
		this.getDataSet().add(index, bag);
	}

	/**
	 * Gets an instance of a bag.
	 * 
	 * @param bagIndex
	 *            The index of the bag of the data set
	 * @param instanceIndex
	 *            Is the index of the instance
	 * @return Instance
	 * @throws IndexOutOfBoundsException
	 */
	public Instance getInstance(int bagIndex, int instanceIndex) throws IndexOutOfBoundsException {
		return this.getDataSet().instance(bagIndex).relationalValue(1).instance(instanceIndex);
	}

	/**
	 * Gets the number of bags of the miml dataset.
	 * 
	 * @return a number of bags
	 */
	public int getNumBags() {
		return this.getNumInstances();
	}

	/**
	 * Gets the number of instances of a bag.
	 * 
	 * @param bag
	 *            A bag of instances
	 * @return int
	 */
	public int getNumInstances(Bag bag) {
		return bag.relationalValue(1).numInstances();
	}

	/**
	 * Gets the number of instances of a bag.
	 * 
	 * @param bagIndex
	 *            A bag index
	 * @return int
	 * @throws Exception Potential exception thrown. To be handled in an upper level.
	 */
	public int getNumInstances(int bagIndex) throws Exception {
		return this.getBag(bagIndex).relationalValue(1).numInstances();
	}

	/**
	 * Gets the number of attributes of the dataset considering relational
	 * attribute with bags as a single attribute.
	 * 
	 * @return int
	 */
	public int getNumAttributes() {
		return this.getDataSet().numAttributes();
	}

	/**
	 * Gets the total number of attributes of the dataset considering all
	 * attributes of bags contained in the relational attribute.
	 * 
	 * @return int
	 */
	public int getNumAttributesWithRelational() {
		return this.getDataSet().numAttributes() + this.getNumAtributtesPerBag() - 1;
	}

	/**
	 * Gets the number of attributes per bag. In miml all bags have the same
	 * number of attributes.
	 * 
	 * @return int
	 */
	public int getNumAtributtesPerBag() {
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
