package mimltransformation;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MLTransformer {
	/**
	 * 
	 * @param train
	 * @return
	 * @throws Exception
	 */
	public Instances geometricTransformation(Instances train) throws Exception{
		
		//Attribute classAttribute =  (Attribute) train.classAttribute().copy();
		Attribute classAttribute = train.attribute(1);
	    Attribute bagLabel = train.attribute(0);
	    double labelValue;

	    Instances newData = train.attribute(1).relation().stringFreeStructure();

	    // insert a bag label attribute at the begining
	    newData.insertAttributeAt(bagLabel, 0);

	    // insert a class attribute at the end
	    newData.insertAttributeAt(classAttribute, newData.numAttributes());
	    newData.setClassIndex(newData.numAttributes() - 1);

	    Instances mini_data = newData.stringFreeStructure();
	    Instances max_data = newData.stringFreeStructure();

	    Instance newInst = new DenseInstance(newData.numAttributes());
	    Instance mini_Inst = new DenseInstance(mini_data.numAttributes());
	    Instance max_Inst = new DenseInstance(max_data.numAttributes());
	    newInst.setDataset(newData);
	    mini_Inst.setDataset(mini_data);
	    max_Inst.setDataset(max_data);

	    double N = train.numInstances();// number of bags
	    for (int i = 0; i < N; i++) {
	    	int attIdx = 1;
	    	Instance bag = train.instance(i); // retrieve the bag instance
	    	labelValue = bag.value(0);
	    	mini_Inst.setValue(0, labelValue);
	    	max_Inst.setValue(0, labelValue);

	    	Instances data = bag.relationalValue(1); // retrieve relational value for
	                                               // each bag
	    	for (int j = 0; j < data.numAttributes(); j++) {
	    		double value;
	    		double[] minimax = minimax(data, j);
	    		value = (minimax[0] + minimax[1]) / 2.0;
	    		newInst.setValue(attIdx++, value);
	        
	    	}
	    	//if (!bag.classIsMissing()) {
	    	//	newInst.setClassValue(bag.classValue()); // set class value
	        //}
	        newData.add(newInst);
	     
	    }
	    
	    
	    return newData;
	}
	/**
	 * 
	 * @param train
	 * @return
	 * @throws Exception
	 */
	public Instances arithmeticTransformation(Instances train) throws Exception {

		//Attribute classAttribute = (Attribute) train.classAttribute().copy();
		Attribute classAttribute = train.attribute(1);
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
		
		double N = train.numInstances();// number of bags
		for (int i = 0; i < N; i++) {
			int attIdx = 1;
			Instance bag = train.instance(i); // retrieve the bag instance
			labelValue = bag.value(0);
			
			Instances data = bag.relationalValue(1); // retrieve relational value for each bag
			for (int j = 0; j < data.numAttributes(); j++) {
				double value;
				value = data.meanOrMode(j);
				newInst.setValue(attIdx++, value);
        
			}
			newData.add(newInst);

		}

    
		return newData;
	}
	
	public Instances arithmeticTransformation(Instance bag) throws Exception {

		//Attribute classAttribute = (Attribute) train.classAttribute().copy();
		Attribute classAttribute = bag.attribute(1);
		Attribute bagLabel = bag.attribute(0);
		double labelValue;
	
		classAttribute.name();
		Instances newData = bag.attribute(1).relation().stringFreeStructure();

		// insert a bag label attribute at the begining
		newData.insertAttributeAt(bagLabel, 0);
		// insert a class attribute at the end
		newData.insertAttributeAt(classAttribute, newData.numAttributes());
		newData.setClassIndex(newData.numAttributes() - 1);

		Instance newInst = new DenseInstance(newData.numAttributes());

		newInst.setDataset(newData);
		
		double N = bag.numAttributes();// number of bags
		for (int i = 0; i < N; i++) {
			int attIdx = 1;
			Instance bagTrain = bag.dataset().get(i); // retrieve the bag instance
			labelValue = bagTrain.value(0);
			
			Instances data = bagTrain.relationalValue(1); // retrieve relational value for each bag
			for (int j = 0; j < data.numAttributes(); j++) {
				double value;
				value = data.meanOrMode(j);
				newInst.setValue(attIdx++, value);
        
			}
			newData.add(newInst);

		}

    
		return newData;
	}

	/**
	 * 
	 * @param train
	 * @return
	 * @throws Exception
	 */
	public Instances MinMaxTransformation(Instances train) throws Exception {

		//Attribute classAttribute = (Attribute) train.classAttribute().copy();
		Attribute classAttribute = train.attribute(1);
		Attribute bagLabel = train.attribute(0);
		double labelValue;
		
		Instances newData = train.attribute(1).relation().stringFreeStructure();

		// insert a bag label attribute at the begining
		newData.insertAttributeAt(bagLabel, 0);
		
		// insert a class attribute at the end
		newData.insertAttributeAt(classAttribute, newData.numAttributes());
		newData.setClassIndex(newData.numAttributes() - 1);
		
		Instances mini_data = newData.stringFreeStructure();
		Instances max_data = newData.stringFreeStructure();

		Instance newInst = new DenseInstance(newData.numAttributes());
		Instance mini_Inst = new DenseInstance(mini_data.numAttributes());
		Instance max_Inst = new DenseInstance(max_data.numAttributes());
		newInst.setDataset(newData);
		mini_Inst.setDataset(mini_data);
		max_Inst.setDataset(max_data);
		
		double N = train.numInstances();// number of bags
		for (int i = 0; i < N; i++) {
			int attIdx = 1;
			Instance bag = train.instance(i); // retrieve the bag instance
			labelValue = bag.value(0);
    
			newInst.setValue(0, labelValue);
      

			Instances data = bag.relationalValue(1); // retrieve relational value for
                                               // each bag
			for (int j = 0; j < data.numAttributes(); j++) {
				double value;
				double[] minimax = minimax(data, j);
				mini_Inst.setValue(attIdx, minimax[0]);// minima value
				max_Inst.setValue(attIdx, minimax[1]);// maxima value
				attIdx++;
				
			}

      
			//if (!bag.classIsMissing()) {
				//max_Inst.setClassValue(bag.classValue()); // set class value
			//}
			mini_data.add(mini_Inst);
			max_data.add(max_Inst);
      
		}

    
		mini_data.setClassIndex(-1);
		mini_data.deleteAttributeAt(mini_data.numAttributes() - 1); // delete
                                                                  // class
                                                                  // attribute
                                                                  // for the
                                                                  // minima data
		max_data.deleteAttributeAt(0); // delete the bag label attribute for the
                                     // maxima data

		//newData = Instances.mergeInstances(mini_data, max_data); // merge minima
                                                               // and maxima
                                                               // data
		newData.setClassIndex(newData.numAttributes() - 1);

    

		return newData;
  }
	/**
	 * 
	 * @param data
	 * @param attIndex
	 * @return
	 */
	public double[] minimax(Instances data, int attIndex) {
		double[] rt = { Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };
	    for (int i = 0; i < data.numInstances(); i++) {
	    	double val = data.instance(i).value(attIndex);
	    	if (val > rt[1]) {
	    		rt[1] = val;
	    	}
	    	if (val < rt[0]) {
	    		rt[0] = val;
	    	}
	    }

	    for (int j = 0; j < 2; j++) {
	    	if (Double.isInfinite(rt[j])) {
	    		rt[j] = Double.NaN;
	    	}
	    }

	    return rt;
	}



}
