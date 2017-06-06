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
package tutorial;

import java.io.File;

import data.MIMLInstances;
import data.MLSaver;
import mulan.data.MultiLabelInstances;
import transformation.mimlTOmil.LPTransformation;
import transformation.mimlTOml.ArithmeticTransformation;
import transformation.mimlTOml.GeometricTransformation;
import transformation.mimlTOml.MiniMaxTransformation;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class for basic handling of the transformation MIML to MIL 
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class exampleMIMLtoMILTranformation {
	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff file.");
		System.out.println("\t-x xmlPathFileName -> path of arff file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar exampleMIMLtoMILTransformation -f data" + File.separator + "toy.arff -x data"
				+ File.separator + "toy.xml");
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {
		// String arffFileName = Utils.getOption("f", args);
		// String xmlFileName = Utils.getOption("x", args);
		// String arffFileName = "data+File.separator+miml_03_data.arff";
		// String xmlFileName = "data+File.separator+miml_03_data.xml";
		// String arffFileName = "data"+File.separator+"miml_text_data_random_80train.arff";
		// String xmlFileName =  "data"+File.separator+"miml_text_data.xml");

		String arffFileName = "data" + File.separator + "toy.arff";
		String xmlFileName = "data" + File.separator + "toy.xml";
		String arffFileResult = "data" + File.separator + "toyResult.arff";
		String xmlFileResult = "data" + File.separator + "toyResult.xml";

		
		// Parameter checking
		if (arffFileName.isEmpty()) {
			System.out.println("Arff pathName must be specified.");
			showUse();
		}
		if (xmlFileName.isEmpty()) {
			System.out.println("Xml pathName must be specified.");
			showUse();
		}

		// Loads the dataset
		System.out.println("Loading the dataset....");
		
		MIMLInstances mimlDataSet =  new MIMLInstances(arffFileName, xmlFileName); 
		LPTransformation lp = new LPTransformation();
		Instances transform = lp.transformBags(mimlDataSet);
		MLSaver.saveArff(transform, arffFileResult);
		MLSaver.saveXml(transform, xmlFileName);
		
		
	}

}
