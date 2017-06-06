/*    This program is free software; you can redistribute it and/or modify
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
package transformation.mimlTOml;

import data.Bag;
import data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Abstract class that prepares the template for a MIMLInstances class to 
 * pass to a MultiLabel class. From it inherited all kinds of transformations. 
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public abstract class MIMLtoML {
	/**
	 * Abstract method to implement in transformation classes. Transform in
	 * a MultiLabel Instances
	 * 
	 * @return a MultiLabel Instances 
	 * @throws Exception Potential exception thrown. To be handled in an upper level.
	 */
	public abstract MultiLabelInstances transformDataset() throws Exception;
	/**
	 * Abstract method to implement in transformation classes. Transform a bag class in
	 * an instance
	 * @param bag
	 * 			a bag of data set that transform in an instance
	 * @return  a transformed instance
	 * @throws Exception Potential exception thrown. To be handled in an upper level.
	 */
	public abstract Instance transformInstance(Bag bag) throws Exception;
	/** array of updated label indices	 */
	protected int updatedLabelIndices[];
	/** Template for save an instances*/
	protected Instances template = null;
	/** Data set for save a MIMLInstances*/ 
	protected MIMLInstances dataset = null;
	/**
	 * Prepare Template for make the transformation MIML to ML.
	 * 
	 * @throws Exception Potential exception thrown. To be handled in an upper level.
	 */
	protected void prepareTemplate() throws Exception {
		int labelIndices[] = dataset.getLabelIndices();
		Instances bags = dataset.getDataSet();

		template = bags.attribute(1).relation().stringFreeStructure();
		// insert a bag label attribute at the begining
		Attribute bagLabel = (Attribute) bags.attribute(0);
		template.insertAttributeAt(bagLabel, 0);

		// Insert labels as attributes in the dataset
		updatedLabelIndices = new int[labelIndices.length];
		for (int i = 0; i < labelIndices.length; i++) {
			Attribute attr = bags.attribute(labelIndices[i]);
			updatedLabelIndices[i] = template.numAttributes();
			template.insertAttributeAt(attr, updatedLabelIndices[i]);
		}
	}

	

	/**
	 * Get the minimal and maximal value of a certain attribute in a certain
	 * data
	 *
	 * @param data
	 *            the data
	 * @param attIndex
	 *            the index of the attribute
	 * @return the double array containing in entry 0 for min and 1 for max.
	 */
	public static double[] minimax(Instances data, int attIndex) {
		double[] rt = { Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };
		for (int i = 0; i < data.numInstances(); i++) {
			double val = data.instance(i).value(attIndex);
			if (val > rt[1])
				rt[1] = val;
			if (val < rt[0])
				rt[0] = val;
		}

		for (int j = 0; j < 2; j++)
			if (Double.isInfinite(rt[j]))
				rt[j] = Double.NaN;

		return rt;
	}
}
