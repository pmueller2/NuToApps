#include "nuto/mechanics/cell/CellIpData.h"
#include "nuto/mechanics/cell/DifferentialOperators.h"
#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/dofs/DofNumbering.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"
#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/base/Timer.h"

#include <iostream>

using namespace NuTo;

/* Attempt to benchmark and optimize the time
 * needed to calculate the right hand side of
 * a simple wave problem
 */
int main(int argc, char *argv[]) {
  int nX = 100;
  int nY = 100;
  int nZ = 10;
  int order = 2;

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  Laws::LinearElastic<3> steel(E, nu);

  MeshFem mesh;
  {
    Timer timer("CreateMesh");
    mesh = UnitMeshFem::CreateBricks(nX, nY, nZ);
  }
  std::cout << std::flush;
  auto allElements = mesh.ElementsTotal();

  DofType dof("displacement", 3);
  InterpolationBrickLobatto ipol(order);
  {
    Timer timer("AddDofInterpolation");
    AddDofInterpolation(&mesh, dof, allElements, ipol);
  }
  std::cout << std::flush;

  auto allCoordinateNodes = mesh.NodesTotal();
  auto allDofNodes = mesh.NodesTotal(dof);

  Constraint::Constraints constraint;

  DofInfo dofInfo = DofNumbering::Build(allDofNodes, dof, constraint);
  SimpleAssembler asmbl(dofInfo);

  IntegrationTypeTensorProduct<3> intType(order + 1,
                                          eIntegrationMethod::LOBATTO);

  CellStorage cells;
  auto cellGroup = cells.AddCells(allElements, intType);

  auto rightHandSide = [dof, &steel](const CellIpData &cipd) {

    DofVector<double> gradient;

    Eigen::MatrixXd B = cipd.B(dof, Nabla::Strain());
    gradient[dof] =
        B.transpose() *
        steel.Stress(cipd.Apply(dof, Nabla::Strain()), 0., cipd.Ids());

    return gradient;
  };

  Eigen::VectorXd localGradient(ipol.GetNumNodes() * dof.GetNum());

  auto constantRHS = [dof, &localGradient](const CellIpData &cipd) {

    DofVector<double> gradient;
    gradient[dof] = localGradient;
    return gradient;
  };

  DofVector<double> globalDofVector;
  {
    Timer timer("VectorAssembly");
    globalDofVector = asmbl.BuildVector(cellGroup, {dof}, constantRHS);
  }
  std::cout << std::flush;
}
