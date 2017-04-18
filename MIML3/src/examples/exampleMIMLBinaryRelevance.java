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

import data.Bag;
import data.MIMLInstances;

import mimlclassifier.MIMLBinaryRelevance;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.Classifier;



public class exampleMIMLBinaryRelevance {

	public static void main(String[] args) {
		try{
			
			//MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_03_data.arff", "data"+File.separator+"miml_03_data.xml");
			
			System.out.println("Loading datasets...");
			
			MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_text_data_random_80train.arff", "data"+File.separator+"miml_text_data.xml");			
			MultiLabelInstances mlTrain = new MultiLabelInstances("data"+File.separator+"miml_text_data_random_80train.arff", "data"+File.separator+"miml_text_data.xml");
			MIMLInstances mimlTest =  new MIMLInstances("data"+File.separator+"miml_text_data_random_20test.arff", "data"+File.separator+"miml_text_data.xml"); 
			MultiLabelInstances mlTest =  new MultiLabelInstances("data"+File.separator+"miml_text_data_random_20test.arff", "data"+File.separator+"miml_text_data.xml"); 
                      
            Classifier  baseLearner = new weka.classifiers.mi.MIBoost();
            
            MIMLBinaryRelevance MIMLBR = new MIMLBinaryRelevance(baseLearner);            
            BinaryRelevance br = new BinaryRelevance(baseLearner);
            
            MIMLBR.setDebug(true);
            MIMLBR.build(mimlTrain);
            
            br.build(mlTrain);
            
            //Prueba 1
            /*Bag bag = mimlTrain.getBag(1);
            MultiLabelOutput prediction = MIMLBR.makePrediction(bag);
            */
            
            /*
            //Prueba 2
            Evaluator eval = new Evaluator();
            MultipleEvaluation results;
            int numFolds = 2;
            results = eval.crossValidate(MIMLBR, mimlTrain, numFolds);
            System.out.println(results);
            */
            
            //Prueba 3
            Evaluator eval2 = new Evaluator();
            Evaluation results2 = eval2.evaluate(MIMLBR, mimlTest, mimlTrain);
            Evaluation results3 = eval2.evaluate(br, mlTest, mlTrain);
            
            System.out.println(results2);
            System.out.println(results3);
            
            
            System.out.println("The program has finished.");
			
		}catch (IndexOutOfBoundsException ioobe){
			System.err.println("Exception: Incorrect index of Bag" );
        } catch (Exception e) {
            e.printStackTrace();
        }
	}

}
