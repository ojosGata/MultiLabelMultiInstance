/*
 *    This program is free software; you can redistribute it and/or modify
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

package data.statistics;

import java.util.HashMap;
import weka.core.Instances;

/**
 * Class with methods to obtain information about a MIL dataset such as the
 * number of attributes per bag, the average number of instances per bag, and
 * the distribution of number of instances per bag...
 * 
 * @author F.J. Gonzalez
 * @author Eva Gigaja
 * @version 20150925
 */
public class MILStatistics {
	/** The minimum number of instances per bag. */
	int minInstancesPerBag;
	/** The maximum number of instances per bag. */
	int maxInstancesPerBag;
	/** The average number of instances per bag. */
	double avgInstancesPerBag;
	/** The number of attributes per bag. */
	int attributesPerBag;
	/** The number of bags. */
	int numBags;
	/** The distribution of number of instances per bag: */
	HashMap<Integer, Integer> distributionBags;

	/**
	 * Calculates various MIML statistics, such as instancesPerBag and
	 * attributesPerBag
	 * 
	 * @param dataSet
	 *            A MIL dataset
	 */
	public void calculateStats(Instances dataSet) {
		numBags = dataSet.numInstances();
		attributesPerBag = dataSet.instance(0).relationalValue(1).numAttributes();
		minInstancesPerBag = Integer.MAX_VALUE;
		maxInstancesPerBag = Integer.MIN_VALUE;

		// Each pair <Integer, Integer> stores <numberOfInstances, numberOfBags>
		distributionBags = new HashMap<Integer, Integer>();
		for (int i = 0; i < numBags; i++) {
			int nInstances = dataSet.instance(i).relationalValue(1).numInstances();
			if (nInstances < minInstancesPerBag) {
				minInstancesPerBag = nInstances;
			}
			if (nInstances > maxInstancesPerBag) {
				maxInstancesPerBag = nInstances;
			}
			if (distributionBags.containsKey(nInstances)) {
				distributionBags.put(nInstances, distributionBags.get(nInstances) + 1);
			} else {
				distributionBags.put(nInstances, 1);
			}
		}

		avgInstancesPerBag = 0.0;
		for (Integer set : distributionBags.keySet()) {
			avgInstancesPerBag += set * distributionBags.get(set);
		}
		avgInstancesPerBag = avgInstancesPerBag / numBags;
	}

	/**
	 * Returns the average number of instances per bag.
	 * 
	 * @return instancesPerBag
	 */
	public double getAvgInstancesPerBag() {
		return avgInstancesPerBag;
	}

	/**
	 * Returns the number of attributes per bag.
	 * 
	 * @return attributesPerBag
	 */
	public int getAttributesPerBag() {
		return attributesPerBag;
	}

	/**
	 * Returns the number of bags.
	 * 
	 * @return numBags
	 */
	public int getnumBags() {
		return numBags;
	}

	/**
	 * Returns the distribution of number of instances per bags.
	 * 
	 * @return distributionBags
	 */
	public HashMap<Integer, Integer> getDistributionBags() {
		return distributionBags;
	}

	/**
	 * Returns distributionBags in textual representation.
	 * 
	 * @return string
	 */
	protected String distributionBagsToString() {
		StringBuilder sb = new StringBuilder();
		for (Integer set : distributionBags.keySet()) {
			sb.append("\n\t<" + distributionBags.get(set) + "," + set + ">");
		}
		return (sb.toString());
	}

	/**
	 * Returns distributionBags in CSV representation.
	 * 
	 * @return string
	 */
	protected String distributionBagsToCSV() {
		StringBuilder sb = new StringBuilder();
		for (Integer set : distributionBags.keySet()) {
			sb.append("\n" + distributionBags.get(set) + ";" + set);
		}
		return (sb.toString());
	}

	/**
	 * Returns statistics in textual representation.
	 * 
	 * @return string
	 */
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\n----------------------------");
		sb.append("\nMIL Statistics--------------");
		sb.append("\n----------------------------");
		sb.append("\nnBags: " + numBags);
		sb.append("\nAvgInstancesPerBag: " + avgInstancesPerBag);
		sb.append("\nMinInstancesPerBag: " + minInstancesPerBag);
		sb.append("\nMaxInstancesPerBag: " + maxInstancesPerBag);
		sb.append("\nAttributesPerBag: " + attributesPerBag);
		sb.append("\nDistribution of bags <nBags, nInstances>:");
		sb.append(distributionBagsToString());
		return (sb.toString());
	}

	/**
	 * Returns statistics in CSV representation.
	 * 
	 * @return string
	 */
	public String toCSV() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nMIL STATISTICS:");
		sb.append("\nnBags;" + numBags);
		sb.append("\nAvgInstancesPerBag;" + avgInstancesPerBag);
		sb.append("\nMinInstancesPerBag;" + minInstancesPerBag);
		sb.append("\nMaxInstancesPerBag;" + maxInstancesPerBag);
		sb.append("\nAttributesPerBag;" + attributesPerBag);
		sb.append("\nDistribution of bags <nBags, nInstances>");
		sb.append(distributionBagsToCSV());
		return (sb.toString());
	}

	/**
	 * Returns the minimum number of instances per bag.
	 * 
	 * @return minInstancesPerBag
	 */
	public int getMinInstancesPerBag() {
		return minInstancesPerBag;
	}

	/**
	 * Returns the maximum number of instances per bag.
	 * 
	 * @return maxInstancesPerBag
	 */
	public int getMaxInstancesPerBag() {
		return maxInstancesPerBag;
	}
}
