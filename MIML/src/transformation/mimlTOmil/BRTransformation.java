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
package transformation.mimlTOmil;
import data.Bag;
import data.MIMLInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class 
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class BRTransformation {

	protected BinaryRelevanceTransformation BRT;

	/**
	 * Constructor
	 * 
	 *
	 * @param data
	 *            a MIMLInstances dataset
	 * @return
	 */
	public BRTransformation(MIMLInstances dataSet) {
		this.BRT = new BinaryRelevanceTransformation(dataSet);
	}

	/**
	 * Remove all label attributes except labelToKeep
	 *
	 * @param instance
	 *            the instance from which labels are to be removed
	 * @param labelToKeep
	 *            the label to keep
	 * @return transformed Instance
	 */
	public Instance transformBag(Bag instance, int labelToKeep) {
		return BRT.transformInstance(instance, labelToKeep);
	}

	/**
	 * Remove all label attributes except labelToKeep
	 *
	 * @param labelToKeep
	 *            the label to keep
	 * @return transformed Instances object
	 * @throws Exception
	 *             when removal fails
	 */
	public Instances transformBags(int labelToKeep) throws Exception {
       return BRT.transformInstances(labelToKeep);
	}

	/**
	 * Remove all label attributes except that at indexOfLabelToKeep
	 *
	 * @param train
	 *            -
	 * @param labelIndices
	 *            -
	 * @param indexToKeep
	 *            the label to keep
	 * @return transformed Instances object
	 * @throws Exception
	 *             when removal fails
	 */
	//Modificación de Instances train por Bag train. Dentro del return he puesto train.dataset()
	public static Instances transformBags(Bag train, int[] labelIndices, int indexToKeep) throws Exception {
	   return BinaryRelevanceTransformation.transformInstances(train.dataset(), labelIndices, indexToKeep);
	}

	/**
	 * Remove all label attributes except label at position indexToKeep
	 *
	 * @param instance
	 *            the instance from which labels are to be removed
	 * @param labelIndices
	 *            the label indices to remove
	 * @param indexToKeep
	 *            the label to keep
	 * @return transformed Instance
	 * 
	 * 
	 */
	public static Instance transformBag(Bag instance, int[] labelIndices, int indexToKeep) {
       return BinaryRelevanceTransformation.transformInstance(instance, labelIndices, indexToKeep);
	}
}
