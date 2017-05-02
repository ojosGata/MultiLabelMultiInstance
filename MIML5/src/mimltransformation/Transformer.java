package mimltransformation;

import data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

public interface Transformer {
	public MultiLabelInstances transformDataset(MIMLInstances dataset);
	public MultiLabelInstances transformInstance(MIMLInstances dataset);
}
