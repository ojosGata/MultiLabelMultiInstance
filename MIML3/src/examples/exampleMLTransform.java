package examples;

import java.io.File;

import data.MIMLInstances;
import mimltransformation.MLTransformer;
import mulan.data.InvalidDataFormatException;
import weka.classifiers.mi.miti.Bag;
import weka.core.Instances;


public class exampleMLTransform {

	public static void main(String[] args) throws Exception {
		System.out.println("Loading datasets...");
		
		//MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_text_data_random_80train.arff", "data"+File.separator+"miml_text_data.xml");			
		MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_03_data.arff", "data"+File.separator+"miml_03_data.xml");
		//MIMLInstances mimlTest =  new MIMLInstances("data"+File.separator+"miml_text_data_random_20test.arff", "data"+File.separator+"miml_text_data.xml"); 
		
		
		MLTransformer transformation=new MLTransformer();
		Instances dataSet=mimlTrain.getDataSet();
		System.out.println(dataSet.numInstances());
		
		System.out.println("=============Arithmetic=====================");
		System.out.println(transformation.arithmeticTransformation(dataSet));
		System.out.println("=============Arithmetic_Instance=====================");
		System.out.println(transformation.arithmeticTransformation(mimlTrain.getBag(1)));
		System.out.println("----------------------------");
		//System.out.println("=============Geometric=====================");
		//System.out.println(transformation.geometricTransformation(dataSet));
		//System.out.println("----------------------------");
		//System.out.println("=============MinMax=====================");
		//System.out.println(transformation.MinMaxTransformation(dataSet));
		//System.out.println("----------------------------");
		
		

	}

}
