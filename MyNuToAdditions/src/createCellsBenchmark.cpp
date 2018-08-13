#include "nuto/base/Timer.h"
#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"
#include "nuto/mechanics/tools/CellStorage.h"
#include <boost/ptr_container/ptr_vector.hpp>
#include <iostream>

int main(int arg, char *argv[]) {
  std::cout << "Create some cells" << std::endl;

  NuTo::Timer t1("mesh");

  NuTo::MeshFem mesh = NuTo::UnitMeshFem::CreateBricks(100, 100, 100);
  NuTo::IntegrationTypeTensorProduct<3> intgr(
      3, NuTo::eIntegrationMethod::LOBATTO);

  t1.Reset("AllElements");
  auto allElements = mesh.ElementsTotal();

  t1.Reset("Cells");
  NuTo::CellStorage cells;
  cells.AddCells(allElements, intgr);

  // Self made
  NuTo::Group<NuTo::CellInterface> cellGroup;
  t1.Reset("MyCells create");
  boost::ptr_vector<NuTo::CellInterface> myCells;
  for (auto &element : allElements) {
    myCells.push_back(new NuTo::Cell(element, intgr, 0));
  }
  t1.Reset("CellGroup create");
  for (auto &c : myCells) {
    cellGroup.Add(c);
  }
}
