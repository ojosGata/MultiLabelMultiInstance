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

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class inheriting from DenseInstance to represent miml bags.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */

public class Bag extends DenseInstance implements Instance{
	
	/**For serialization*/
	private static final long serialVersionUID = 1L;   

    
	/**Constructor.
	 * @param instance
	 * 				The instance become a Bag
	 */
	public Bag(Instance instance) throws Exception {	    
	    super(instance);
	    m_AttValues = instance.toDoubleArray();
	    m_Weight = instance.weight();	        
	    m_Dataset = instance.dataset();	  
	}	
	
	/**
	 * Returns an instance of the bag.
	 * 
	 * @param  bagIndex
	 * 			The number of index 
	 * @return Instance 
	 * 			
	 */
	public Instance getInstance(int bagIndex)
	{		
		return this.relationalValue(1).instance(bagIndex);
	}
	
	/**
	 * Gets the total number of attributes of the bag considering all attributes contained in the relational attribute.
	 * 
	 * @return int
	 */
	public int getNumAttributesWithRelational()
	{
		return this.numAttributes()+this.relationalValue(1).numAttributes()-1;
	}
	
	/**
	 * Gets the number of attributes of an instance in the bag.* 
	 * @return int
	 */
	public int getNumAttributesInABag()
	{
		return this.relationalValue(1).numAttributes();
	}
	
	/**
	 * Gets the number of instances of the bag.
	 * 
	 * @return int
	 */	
	public int getNumInstances()
	{
		return this.relationalValue(1).numInstances();		
	}

	/**
	 * Gets a bag in the form of a set of instances considering just the relational information.
	 * Identification of bag and information about labels is not included.
	 * 
	 * @return Instances
	 * @throws Exception Potential exception thrown. To be handled in an upper level.
	 */
	public Instances getBagAsInstances() throws Exception {				 
			Instances bags = this.relationalValue(1);
			return bags;		
	}
	
}
