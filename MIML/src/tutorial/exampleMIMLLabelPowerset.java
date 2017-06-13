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

import data.Bag;
import data.MIMLInstances;
import mimlclassifier.MIMLLabelPowerset;
import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.Classifier;
import weka.classifiers.mi.MISMO;

/**
 * 
 * Class for basic handling of the classifier {@link MIMLLabelPowerset}.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class exampleMIMLLabelPowerset {
	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff file.");
		System.out.println("\t-x xmlPathFileName -> path of arff file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar exampleMIMLLabelPowerset -f data" + File.separator + "toy.arff -x data"
				+ File.separator + "toy.xml");
		System.exit(-1);
	}

	public static void main(String[] args) {

		try {
			// String arffFileNameTrain = Utils.getOption("f", args);
			// String xmlFileName = Utils.getOption("x", args);

			String arffFileNameTrain = "data" + File.separator + "miml_text_data_random_80train.arff";
			String arffFileNameTest = "data" + File.separator + "miml_text_data_random_20test.arff";
			String xmlFileName = "data" + File.separator + "miml_text_data.xml";

			// Parameter checking
			if (arffFileNameTrain.isEmpty()) {
				System.out.println("Arff pathName must be specified.");
				showUse();
			}
			if (arffFileNameTest.isEmpty()) {
				System.out.println("Arff pathName must be specified.");
				showUse();
			}
			if (xmlFileName.isEmpty()) {
				System.out.println("Xml pathName must be specified.");
				showUse();
			}

			// Loads the dataset
			System.out.println("Loading the dataset....");

			MIMLInstances mimlTrain = new MIMLInstances(arffFileNameTrain, xmlFileName);
			MIMLInstances mimlTest = new MIMLInstances(arffFileNameTest, xmlFileName);

			Classifier baseLearner = new MISMO();
			MIMLLabelPowerset MIMLLP = new MIMLLabelPowerset(baseLearner);

			MIMLLP.setDebug(true);
			MIMLLP.build(mimlTrain);

			// Evaluates a single instance
			Bag bag = mimlTrain.getBag(1);
			MultiLabelOutput prediction = MIMLLP.makePrediction(bag);
			System.out.println("\nPrediction on a single instance:\n\t" + prediction.toString());

			// Performs a train-test evaluation
			Evaluator evalTT = new Evaluator();
			System.out.println("\nPerforming train-test evaluation:\n");
			Evaluation resultsTT = evalTT.evaluate(MIMLLP, mimlTest, mimlTrain);
			System.out.println("\nResults on train test evaluation:\n" + resultsTT);

			System.out.println("The program has finished.");

		} catch (IndexOutOfBoundsException ioobe) {
			System.err.println("Exception: Incorrect index of Bag");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
