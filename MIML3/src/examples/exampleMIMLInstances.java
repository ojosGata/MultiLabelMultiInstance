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



package examples;


import java.io.File;

import data.MIMLInstances;
import weka.core.Instance;
import weka.core.Instances;
public class exampleMIMLInstances {

	public static void main(String[] args) {
		try {
            
			
            System.out.println("Loading the dataset");
            
            MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_03_data.arff", "data"+File.separator+"miml_03_data.xml");
            Instances data = mimlTrain.getDataSet();
            System.out.println("The number of bag is " + mimlTrain.getNumBags(data));
            Instance inst1 = mimlTrain.getBag(1);
            System.out.println("Number of Instances  " + mimlTrain.getNumInstances(inst1));
            System.out.println("Number of Attributes "+ mimlTrain.getNumAtributtes(inst1));
            Instance inst2 = mimlTrain.getInstance(data, 1, 4);
            for(int i=0;i<inst2.numAttributes();i++){
            	
            	System.out.println("Attribute "+i+": "+inst2.value(i));
            }
            	
            for(int i=0; i<data.numInstances();i++){
            	System.out.println("Instances " +i);
            	for(int j=0; j<data.get(i).numAttributes();j++){
            		System.out.print("Attribute Value: ");
            		System.out.println(mimlTrain.getInstance(data, i, j));
            	}
			}
		} catch (IndexOutOfBoundsException ioobe){
			System.err.println("Exception: Incorrect index of bag" );
        } catch (Exception e) {
            e.printStackTrace();
        }

	}

}
