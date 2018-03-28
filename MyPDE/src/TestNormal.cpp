#include "nuto/mechanics/cell/Jacobian.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

using namespace NuTo;

int main(int argc, char *argv[]) {

  MeshGmsh gmsh("cube1.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto outerBoundary = gmsh.GetPhysicalGroup("OuterBoundary");
  auto bottomOuterRing = gmsh.GetPhysicalGroup("BottomOuterRing");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  for (auto &elmColl : top) {
    ElementFem &cElm = elmColl.CoordinateElement();
    Eigen::VectorXd nodeVals = cElm.ExtractNodeValues();
    for (int i = 0; i < cElm.GetNumNodes(); i++) {
      Eigen::VectorXd localCoords = cElm.Interpolation().GetLocalCoords(i);
      Eigen::MatrixXd der = cElm.GetDerivativeShapeFunctions(localCoords);
      NuTo::Jacobian jac(nodeVals, der, 3);
      Eigen::VectorXd nrm = jac.Normal();
      std::cout << "Nr " << i << " \n";
      std::cout << "Normal:   " << nrm[0] << ",  " << nrm[1] << ",  " << nrm[2]
                << "\n";
      Eigen::VectorXd coords = cElm.GetNMatrix(localCoords) * nodeVals;
      std::cout << "Coords:   " << coords[0] << ",  " << coords[1] << ",  "
                << coords[2] << "\n";
      std::cout << "---------------------" << std::endl;
    }
  }
}
