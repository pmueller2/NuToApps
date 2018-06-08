#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include <eigen3/unsupported/Eigen/SparseExtra>
#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  MeshGmsh gmsh("plateH0.3_angle45_L0.4small.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto domain = gmsh.GetPhysicalGroup("domain");

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  int order = 1;
  int integrationOrder = order + 1;

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  auto &ipol3D = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol3D);

  auto domainIntType = CreateLobattoIntegrationType(
      domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage domainCells;
  auto domainCellGroup = domainCells.AddCells(domain, *domainIntType);

  DofInfo dofInfo = DofNumbering::Build(mesh.NodesTotal(dof1), dof1,
                                        Constraint::Constraints());

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  std::cout << "NumDofs: " << dofInfo.numIndependentDofs[dof1] << std::endl;

  Eigen::SparseMatrix<double> stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.Hessian0(cipd, 0.);
      })(dof1, dof1);

  std::cout << "NumNonZeros: " << stiffnessMx.nonZeros() << std::endl;
  Eigen::saveMarket(stiffnessMx, "plateH0.3_angle45_L0.4small.mtx");
}
