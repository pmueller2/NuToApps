#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "boost/filesystem.hpp"
#include <iostream>

#include "nuto/base/Timer.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"

using namespace NuTo;

double smearedStepFunction(double t, double tau) {
  double ot = M_PI * t / tau;
  if (ot > M_PI)
    return 1.;
  if (ot < 0.)
    return 0.;
  return 0.5 * (1. - cos(ot));
}

int main(int argc, char *argv[]) {

  // *********************************
  //      Solve
  // *********************************

  double riseTime = 0.200e-6;
  double timeStep = 0.020e-6;
  int numSteps = 2;
  double simulationTime = timeStep * numSteps;

  // *********************************
  //      Geometry parameter
  // *********************************

  MeshFem mesh = UnitMeshFem::Transform(
      UnitMeshFem::CreateBricks(150, 5, 150), [](Eigen::Vector3d x) {
        return Eigen::Vector3d(x[0] * 0.6, x[1] * 0.02, x[2] * 0.6);
      });

  auto domain = mesh.ElementsTotal();

  // **************************************
  // Result directory, filesystem
  // **************************************

  std::string resultDirectory = "/PointForce3D/";
  bool overwriteResultDirectory = true;

  // delete result directory if it exists and create it new
  boost::filesystem::path rootPath = boost::filesystem::initial_path();
  boost::filesystem::path resultDirectoryFull = rootPath.parent_path()
                                                    .parent_path()
                                                    .append("/results")
                                                    .append(resultDirectory);

  if (boost::filesystem::exists(resultDirectoryFull)) // does p actually exist?
  {
    if (boost::filesystem::is_directory(resultDirectoryFull)) {
      if (overwriteResultDirectory) {
        boost::filesystem::remove_all(resultDirectoryFull);
        boost::filesystem::create_directory(resultDirectoryFull);
      }
    }
  } else {
    boost::filesystem::create_directory(resultDirectoryFull);
  }

  // **************************************
  // Dofs, etc.
  // **************************************

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  int order = 2;

  std::cout << "Create Mesh" << std::endl;
  auto &ipol3D = mesh.CreateInterpolation(InterpolationBrickLobatto(order));

  std::cout << "Add Dof Interpolation" << std::endl;
  {
    Timer timer("AddDofInterpolation");
    AddDofInterpolation(&mesh, dof1, domain, ipol3D);
  }

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  std::cout << "Set Dirichlet Boundary" << std::endl;

  Group<NodeSimple> nodesX0 = mesh.NodesAtAxis(eDirection::X, dof1, 0.);
  Group<NodeSimple> nodesZ0 = mesh.NodesAtAxis(eDirection::Z, dof1, 0.);

  Constraint::Constraints constraints;
  constraints.Add(dof1, Constraint::Component(nodesX0, {eDirection::X}));
  constraints.Add(dof1, Constraint::Component(nodesZ0, {eDirection::Z}));

  // ******************************
  //    Load
  // ******************************

  NodeSimple forceNode = mesh.NodeAtCoordinate(Eigen::Vector3d(0.0, 0.02, 0.0));

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  std::cout << "Set up cell group" << std::endl;

  int integrationOrder = order + 1;

  // Domain cells
  auto domainIntType = CreateLobattoIntegrationType(
      domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage domainCells;
  auto domainCellGroup = domainCells.AddCells(domain, *domainIntType);

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  int numDofs =
      dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

  std::cout << "Build C-Matrix" << std::endl;

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildUnitConstraintMatrix(dof1, numDofs);

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  std::cout << "Assemble mass mx" << std::endl;

  DofVector<double> lumpedMassMx;
  {
    Timer timer("AddDofInterpolation");

    lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
        domainCellGroup, {dof1},
        [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });
  }
  std::cout << "Modify mass mx" << std::endl;

  Eigen::SparseMatrix<double> massMxModFull =
      cmat.transpose() * lumpedMassMx[dof1].asDiagonal() * cmat;
  Eigen::VectorXd massMxMod = massMxModFull.diagonal();

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  std::cout << "Assemble stiffness mx" << std::endl;

  DofMatrixSparse<double> stiffnessMx;
  {
    Timer timer("AddDofInterpolation");
    stiffnessMx =
        asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.Hessian0(cipd, 0.);
        });
  }

  std::cout << "Modify stiffness mx" << std::endl;

  Eigen::SparseMatrix<double> stiffnessMxMod =
      cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

  // ***********************************
  //    Solve
  // ***********************************
}
