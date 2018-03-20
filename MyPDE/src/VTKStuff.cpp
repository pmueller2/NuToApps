#include "visualize/UnstructuredGrid.h"
#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {
  Visualize::UnstructuredGrid grid;

  // Quad geometry
  std::vector<int> pIds;
  pIds.push_back(grid.AddPoint(Eigen::Vector2d(0., 0.)));
  pIds.push_back(grid.AddPoint(Eigen::Vector2d(1., 0.)));
  pIds.push_back(grid.AddPoint(Eigen::Vector2d(1., 1.)));
  pIds.push_back(grid.AddPoint(Eigen::Vector2d(0., 1.)));

  grid.AddCell(pIds, eCellTypes::QUAD);

  // Point data
  std::string name = "Data";
  grid.DefinePointData(name);
  for (int pId : pIds)
    grid.SetPointData(pId, name, Eigen::Vector2d(1.2, 2.3));

  grid.ExportVtuDataFile("VTKTestAscii.vtu", false);
  grid.ExportVtuDataFile("VTKTestBinary.vtu", true);
}
