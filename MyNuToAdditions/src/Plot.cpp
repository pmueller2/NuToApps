#include <iostream>
#include "../../NuToHelpers/VisualizeTools.h"
using namespace NuTo;

//! Plots points x with data y called dataName. Outputs result in
//! vtu file filename, optional in ascii (asBinary=false)
//! @param x Mx(spaceDim, numPoints)
//! @param y Mx(dataDim, numPoints)
void Plot(Eigen::MatrixXd x, Eigen::MatrixXd y, std::string dataName, std::string fileName, bool asBinary = true)
{
    if (x.cols() != y.cols())
    {
        throw Exception(__PRETTY_FUNCTION__,"cols(x) != cols(y)");
    }

    Visualize::UnstructuredGrid grid;

    // Fill grid
    grid.DefinePointData(dataName);
    for (int i=0; i<x.cols(); i++)
    {
        int pId = grid.AddPoint(x.col(i));
        grid.SetPointData(pId,dataName,y.col(i));
    }

    // Export file
    std::string filename = "TestPlot.vtu";
    grid.ExportVtuDataFile(filename, asBinary);
}

int main(int argc, char *argv[]) {

  // Data to plot
  int numPoints = 10;
  int spaceDim = 3;

  Eigen::MatrixXd points(spaceDim,numPoints);
  Eigen::MatrixXd scalarData(1,numPoints);

  for (int i=0; i<numPoints; i++)
  {
      points(0,i) = i;
      points(1,i) = 2.*i;
      points(2,i) = 0;
      scalarData(0,i) = sqrt(i);
  }

  std::string dataName = "scalarData";
  std::string filename = "TestPlot.vtu";

  Visualize::Plot(points,scalarData,dataName,filename);
}
