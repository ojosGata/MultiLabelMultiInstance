package mimltransformation;

import data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ArithmeticTransformation implements Transformer {

	@Override
	public MultiLabelInstances transformDataset(MIMLInstances dataset) {
		MIMLInstances newMIMLInstance = dataset;
		
		Instances train = dataset.getDataSet();
		Attribute classAttribute = (Attribute) train.classAttribute().copy();
		//Attribute classAttribute = dataset.attribute(1);
		Attribute bagLabel = train.attribute(0);
		double labelValue;
		
		classAttribute.name();
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

		return newMIMLInstance;

		
	}

	@Override
	public MultiLabelInstances transformInstance(MIMLInstances dataset) {
		// TODO Auto-generated method stub
		return null;
	}

}
