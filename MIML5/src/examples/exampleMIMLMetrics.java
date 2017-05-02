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


package examples;
import java.io.File;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import statistics.MILStatistics;
import statistics.MLStatistics;
import weka.core.Utils;

/** Class to compute statistics on a multi-instance multi-label dataset.
 * 
 * @author F.J. Gonzalez
 * @author Eva Gigaja
 * @version 20150925 */
public class exampleMIMLMetrics
  {
  /** Shows the help on command line. */
  public static void showUse()
    {
    System.out.println("Program parameters:");
    System.out.println("  -name filename -> file name without extension, arff file name and xml file name must be the same.");
    System.out.println("Examples:");
    if (System.getProperty("os.name").toLowerCase().indexOf("win") >= 0)
      {
      System.out.println("java MIMLMetrics -name \".\\Data\\filename\"");
      }
    else
      {
      System.out.println("java MIMLMetrics -name \"./Data/filename\"");
      }
    System.exit(-1);
    }

  /** Compute statistics on a multi-instance multi-label dataset.
   *
   * @param args
   *          Program parameters:<br>
   *          &nbsp;&nbsp;&nbsp;&nbsp;-name filename<br>
   *          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;File name without
   *          extension, arff file name and xml file name must be the same.<br>
   *          Examples:<br>
   *          &nbsp;&nbsp;&nbsp;&nbsp;java MIMLMetrics -name ".\Data\filename"<br>
   * @throws mulan.data.InvalidDataFormatException */
  public static void main(String[] args) throws InvalidDataFormatException
    {
    try
      {
      // Example of command line argument: -name data/miml_02_miml_text_data
      String name = Utils.getOption("name", args);
      System.out.println("Loading the dataset " + name);
      //MultiLabelInstances dataset = new MultiLabelInstances(name + ".arff", name + ".xml");
      MultiLabelInstances dataset = new MultiLabelInstances("data"+File.separator+"toy.arff", "data"+File.separator+"toy.xml");
      // MLL or MIML
      MLStatistics statsML = new MLStatistics();
      statsML.calculateStats(dataset);
      System.out.println(statsML.toString());
      System.out.print("\n----------------------------");
      System.out.print("\nCoocurrences - dependences  ");
      System.out.print("\n----------------------------");
      statsML.calculateCoocurrence(dataset);
      System.out.println(statsML.coocurrenceToString());
      statsML.calculatePhiChi2(dataset);
      System.out.print("\nPhi correlation matrix");
      System.out.println(statsML.correlationsToString(statsML.getPhi()));
      System.out.print("\nChi correlation matrix");
      System.out.println(statsML.correlationsToString(statsML.getChi2()));
      System.out.println("\nPhi diagram: ");
      statsML.printPhiDiagram(0.05);

      // MIL or MIML
      MILStatistics statsMIL = new MILStatistics();
      statsMIL.calculateStats(dataset.getDataSet());
      System.out.println(statsMIL.toString());
      }
    catch (Exception e)
      {
      e.printStackTrace();
      showUse();
      }
    }
  }
