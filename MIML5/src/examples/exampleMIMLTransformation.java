package examples;

import java.io.File;

import data.MIMLInstances;
import mimltransformation.ArithmeticTransformation;
import mimltransformation.GeometricTransformation;
import mimltransformation.MLTransformer;
import mimltransformation.MinMaxTransformation;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instances;

public class exampleMIMLTransformation {

	public static void main(String[] args) throws Exception {
		System.out.println("Loading datasets...");
		
		//MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_text_data_random_80train.arff", "data"+File.separator+"miml_text_data.xml");			
		MIMLInstances mimlTrain =  new MIMLInstances("data"+File.separator+"miml_04_data.arff", "data"+File.separator+"miml_04_data.xml");
		//MIMLInstances mimlTest =  new MIMLInstances("data"+File.separator+"miml_text_data_random_20test.arff", "data"+File.separator+"miml_text_data.xml"); 
		ArithmeticTransformation ari = new ArithmeticTransformation();
		GeometricTransformation geo = new GeometricTransformation();
		MinMaxTransformation minMax = new MinMaxTransformation();
		Instances dataSet=mimlTrain.getDataSet();
		System.out.println(dataSet.numInstances());
		//ari.transformDataset(mimlTrain);
		MultiLabelInstances result = ari.transformDataset(mimlTrain);
		System.out.println("=============Arithmetic=====================");
		System.out.println(result.getDataSet());
		//System.out.println("=============Arithmetic_Instance=====================");
		//System.out.println(transformation.arithmeticTransformation(mimlTrain.getBag(1)));
		//System.out.println("----------------------------");
		//System.out.println("=============Geometric=====================");
		//System.out.println(geo.transformDataset(mimlTrain).getDataSet());
		//System.out.println("----------------------------");
		//System.out.println("=============MinMax=====================");
		//System.out.println(minMax.transformDataset(mimlTrain).getDataSet());
		//System.out.println("----------------------------");
		
	}

}
