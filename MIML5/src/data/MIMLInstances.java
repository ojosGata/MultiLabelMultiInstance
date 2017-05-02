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
import statistics.MILStatistics;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import data.Bag;
/**
 * 
 * Class inheriting from the MultiLabelnstances class to encompass how to 
 * solve the problem by choosing MultiLabel or MultiInstance
 * 
 * @author Ana I.Reyes Melero
 * @version 20170213
 *
 */
public class MIMLInstances extends MultiLabelInstances{
	
	//Constructors
	public MIMLInstances(Instances data, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(data, xmlLabelsDefFilePath);
		
	}
	/**
	 * 
	 * @param dataSet Set of data
	 * @param labelsMetaData Label set of dataset attributes 
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(Instances dataSet, LabelsMetaData labelsMetaData) throws InvalidDataFormatException {
		super(dataSet, labelsMetaData);
	}
		
	/**
	 * 
	 * @param arffFilePath Path of the .arff file
	 * @param xmlLabelsDefFilePath Path of the .xml file
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(String arffFilePath, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(arffFilePath, xmlLabelsDefFilePath);
	
	}
	
	/**
	 * 
	 * @param arffFilePath Path of the .arff file
	 * @param numLabelAttributes Number of Attributes 
	 * @throws InvalidDataFormatException
	 */
	public MIMLInstances(String arffFilePath, int numLabelAttributes) throws InvalidDataFormatException {
		super(arffFilePath, numLabelAttributes);
	}



		/**
	 * Gets a bag or pattern from MultiLabel instances from a index.	 * 
	 * @param bagIndex index of bag
	 * @return a bag or an instance from the index of the dataset
	 * @throws Exception 
	 */
	public Bag getBag(int bagIndex) throws Exception{
		
		Instances aux = this.getDataSet();
		DenseInstance aux1 = (DenseInstance)aux.get(bagIndex);
		return  new Bag(aux1);
	}
	
	
	/**
	 * Add a bag instance to dataSet
	 * @param dataSet set of data 
	 * @param bag instance of data
	 */
	public void addBag(Instances dataSet, Instance bag){
		dataSet.add(bag);
	}
	/**
	 * Add an instance to a data set 
	 * @param dataSet
	 * @param bag
	 * @param index
	 */
	public void addInstance(Instances dataSet, Instance bag, int index){
		dataSet.add(index, bag);
	}
	
	/**
	 * Get an instance of a dataset bag
	 * @param dataSet is the set of data from which an instance
	 * @param bagIndex Is the index of the bag of the data set.
	 * @param instanceIndex Is the index of the instance that we are going to return.
	 * @return A concrete instance of the dataset
	 * @throws IndexOutOfBoundsException
	 */
	public Instance getInstance(Instances dataSet, int bagIndex, int instanceIndex) throws IndexOutOfBoundsException{
		return dataSet.instance(bagIndex).relationalValue(1).instance(instanceIndex);
	}
	
	
	/**
	 * Get the number of bags that have an instance
	 * @param dataSet Is the dataset with several bags
	 * @return a number of bag
	 */
	public int getNumBags(Instances dataSet){
		return dataSet.numInstances();
	}
	/**
	 * Get the number of instances of a bag...---...
	 * @param bag Is the bag of instances
	 * @return a number of instances
	 */
	public int getNumInstances(Instance bag){
		return bag.relationalValue(1).numInstances();
	}
	/**
	 * Gets the number of attributes of an instance
	 * @param bag Is the bag of instances
	 * @return a number of attributes
	 */
	public int getNumAtributtes(Instance bag){
		return bag.relationalValue(1).numAttributes();
	}
	
	
	/**
	 * Shows the statistics of a multiLabel Instance
	 * @param dataSet Set of data 
	 * @return the statistics
	 */
	public MILStatistics mimlAvgInstancesPerBag(Instances dataSet){
		MILStatistics statistics = new MILStatistics();
		statistics.calculateStats(dataSet);	
		return statistics;
	}
	
	public MultiLabelInstances getMLDataSet(){
		
		return (MultiLabelInstances)this;
	}
	public void setDataset(Instances newData) throws InvalidDataFormatException {
		new MIMLInstances(newData,this.getLabelsMetaData()); 
		
	}
	
}
