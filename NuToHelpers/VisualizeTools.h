#pragma once
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/base/Exception.h"
#include <Eigen/Dense>

namespace NuTo
{
namespace Visualize
{

//! Plots points x with data y called dataName. Outputs result in
//! vtu file filename, optional in ascii (asBinary=false)
//! @param x Mx(spaceDim, numPoints)
//! @param y Mx(dataDim, numPoints)
static void Plot(const Eigen::MatrixXd x, const Eigen::MatrixXd y, const std::string dataName, const std::string fileName, bool asBinary = true)
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

} /* Visualize */
} /* NuTo */

