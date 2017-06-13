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
 * Abstract class to transform MIMLInstances into MultiLabelInstances.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public abstract class MIMLtoML {

	/** Array of updated label indices */
	protected int updatedLabelIndices[];
	/** Template to store Instances */
	protected Instances template = null;
	/** Original data set of MIMLInstances */
	protected MIMLInstances dataset = null;

	/**
	 * Transforms {@link MIMLInstances} into MultiLabelInstances.
	 * 
	 * @return MultiLabelInstances.
	 * @throws Exception
	 *             To be handled in an upper level.
	 */
	public abstract MultiLabelInstances transformDataset() throws Exception;

	/**
	 * Transforms {@link Bag} into Instance.
	 * 
	 * @param bag
	 *            The Bag to be transformed.
	 * @return Instance
	 * @throws Exception
	 *             To be handled in an upper level.
	 */
	public abstract Instance transformInstance(Bag bag) throws Exception;

	/**
	 * Prepares a template to perform the transformation from MIMLInstances to
	 * MultiLabelInstances. This template includes: the bag label attribute, all
	 * attributes in the relational attribute as independent attributes and
	 * label attributes. For instance, in the relation above, the resulting
	 * template is showed.
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
	 * &#064;relation template<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * @throws Exception
	 *             To be handled in an upper level.
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
	 * Get the minimal and maximal value of a certain attribute in a data set.
	 *
	 * @param data
	 *            The data set.
	 * @param attIndex
	 *            The index of the attribute.
	 * @return double[] containing in position 0 the min value and in position 1
	 *         the max value.
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
