package mimltransformation;

import data.MIMLInstances;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ArithmeticTransformation implements Transformer {

	@Override
	public Instances transformDataSet(MIMLInstances dataset) {
		
	
		if(dataset.getDataSet().classIndex() == -1){
			dataset.getDataSet().setClassIndex(dataset.getDataSet().numAttributes() -1);
		}
		Instances train = dataset.getDataSet();
		Attribute classAttribute = (Attribute) train.classAttribute().copy();
		Attribute bagLabel = train.attribute(0);
		double labelValue;
		
		Instances newData = train.attribute(1).relation().stringFreeStructure();

		// insert a bag label attribute at the begining
		newData.insertAttributeAt(bagLabel, 0);
		// insert a class attribute at the end
		newData.insertAttributeAt(classAttribute, newData.numAttributes());
		newData.setClassIndex(newData.numAttributes() - 1);
		Instance newInst = new DenseInstance(newData.numAttributes());
		newInst.setDataset(newData);		
		double N = train.numAttributes();// number of bags
		for (int i = 0; i < N; i++) {
			int attIdx = 1;
			Instance bagTrain = train.get(i); // retrieve the bag instance
			labelValue = bagTrain.value(0);
					
			Instances data = bagTrain.relationalValue(1); // retrieve relational value for each bag
			for (int j = 0; j < data.numAttributes(); j++) {
				double value;
				value = data.meanOrMode(j);
				newInst.setValue(attIdx, value);
				attIdx++;
				
			}
			newData.add(newInst);
			
		}
		//System.out.println();
		return newData;
		
		//newMIMLInstance.add(newInst); 
		
		
		

		
	}

	@Override
	public MultiLabelInstances transformDataset(MIMLInstances dataset) {
		
	
		if(dataset.getDataSet().classIndex() == -1){
			dataset.getDataSet().setClassIndex(dataset.getDataSet().numAttributes() -1);
		}
		MultiLabelInstances newMIMLInstance =dataset;
		Instances train = dataset.getDataSet();
		Attribute classAttribute = (Attribute) train.classAttribute().copy();
		Attribute bagLabel = train.attribute(0);
		double labelValue;
		
		
		Instances newData = train.attribute(1).relation().stringFreeStructure();

		// insert a bag label attribute at the begining
		newData.insertAttributeAt(bagLabel, 0);
		// insert a class attribute at the end
		newData.insertAttributeAt(classAttribute, newData.numAttributes());
		newData.setClassIndex(newData.numAttributes() - 1);
		Instance newInst = new DenseInstance(newData.numAttributes());
		newInst.setDataset(newData);		
		double N = train.numAttributes();// number of bags
		for (int i = 0; i < N; i++) {
			int attIdx = 1;
			Instance bagTrain = train.get(i); // retrieve the bag instance
			labelValue = bagTrain.value(0);
					
			Instances data = bagTrain.relationalValue(1); // retrieve relational value for each bag
			for (int j = 0; j < data.numAttributes(); j++) {
				double value;
				value = data.meanOrMode(j);
			
				newInst.setValue(attIdx++, value);
				
			}
			newData.add(newInst);
			
		}
		
		dataset.addBag(newData, newInst);
		
		newMIMLInstance = dataset;//newData;
		
		//return newData;
		
		//newMIMLInstance.add(newInst); 
		return newMIMLInstance;
		
		

		
	}

	
	@Override
	public MultiLabelInstances transformInstance(MIMLInstances dataset) {
		
		return null;
	}

}
