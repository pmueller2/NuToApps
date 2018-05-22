#include "../../NuToHelpers/MeshValuesTools.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"
#include "nuto/mechanics/tools/CellStorage.h"

#include <iostream>

using namespace NuTo;

void TestGetLocalCoordinatesFromGlobal() {
  MeshGmsh gmsh("plateWithInternalCrackHexed.msh");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  for (ElementCollectionFem &elmColl : domain) {
    ElementFem &elm = elmColl.CoordinateElement();
    Eigen::Vector3d localTestCoord(1.1, -1.2, 1.4);
    Eigen::VectorXd globalTestCoord = Interpolate(elm, localTestCoord);
    auto xi = NuTo::Tools::GetLocalCoordinatesFromGlobal(globalTestCoord, elm);
    std::cout << xi[0] << ", " << xi[1] << ", " << xi[2] << std::endl;
    std::cout << "------------------------" << std::endl;
  }
}

void TestGetNodeElementMap() {
  MeshFem mesh = UnitMeshFem::CreateLines(10);
  auto domain = mesh.ElementsTotal();
  auto nodeElementMap = NuTo::Tools::GetNodeCoordinateElementMap(domain);

  NodeSimple &nd4 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.4));

  auto nearbyelements = nodeElementMap.at(&nd4);

  for (ElementCollectionFem *e : nearbyelements) {
    std::cout << "---------------" << std::endl;
    std::cout << e->CoordinateElement().ExtractNodeValues() << std::endl;
  }
}

void TestInterpolator() {
  MeshFem mesh = UnitMeshFem::CreateLines(10);
  Eigen::MatrixXd coords(4, 1);
  coords << 0., 0.05, 0.11, 0.96;
  DofType dof("scalar", 1);
  AddDofInterpolation(&mesh, dof);
  auto &nd0 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.0), dof);
  auto &nd1 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.1), dof);

  nd0.SetValue(0, 7.);
  nd1.SetValue(0, 6.);

  Tools::Interpolator interpolator(coords, mesh.ElementsTotal());
  std::cout << interpolator.GetValue(0, dof) << std::endl;
  std::cout << interpolator.GetValue(1, dof) << std::endl;
  std::cout << interpolator.GetValue(2, dof) << std::endl;
}

int main(int argc, char *argv[]) {
  TestGetLocalCoordinatesFromGlobal();
  // TestGetNodeElementMap();
  // TestInterpolator();
}
