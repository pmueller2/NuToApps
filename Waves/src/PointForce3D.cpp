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

#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

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

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  std::array<int, 3> numElements{60, 2, 60};
  std::array<double, 3> blockDimensions{0.6, 0.02, 0.6};
  int order = 2;

  double riseTime = 0.200e-6;
  double timeStep = 0.020e-6;
  int numSteps = 20000;

  // *********************************
  //      Geometry parameter
  // *********************************

  MeshFem mesh = UnitMeshFem::Transform(
      UnitMeshFem::CreateBricks(numElements[0], numElements[1], numElements[2]),
      [&](Eigen::Vector3d x) {
        return Eigen::Vector3d(x[0] * blockDimensions[0],
                               x[1] * blockDimensions[1],
                               x[2] * blockDimensions[2]);
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

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

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

  NodeSimple &forceNode =
      mesh.NodeAtCoordinate(Eigen::Vector3d(0.0, 0.02, 0.0), dof1);

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

  std::cout << "Num Dofs: " << numDofs << std::endl;

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildUnitConstraintMatrix(dof1, numDofs);

  //  auto jknum = constraints.GetJKNumbering(dof1, numDofs);
  //  Eigen::VectorXi reverseJKNumbering =
  //      ((Eigen::PermutationMatrix<Eigen::Dynamic>(jknum.mIndices)).transpose())
  //          .eval()
  //          .indices();
  //  int forceNodeId = reverseJKNumbering(forceNode.GetDofNumber(1));
  int forceNodeId = forceNode.GetDofNumber(1);

  //  std::cout << "ForceNode global" << forceNode.GetDofNumber(1) << std::endl;
  //  std::cout << "ForceNode independent" << forceNodeId << std::endl;
  //  std::cout << "JK \n" << jknum.mIndices << std::endl;
  //  std::cout << "ReverseJK \n" << reverseJKNumbering << std::endl;

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  std::cout << "Assemble mass mx" << std::endl;

  DofVector<double> lumpedMassMx;
  {
    Timer timer("AssembleMass");

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
    Timer timer("Assemble stiffness");
    stiffnessMx =
        asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.Hessian0(cipd, 0.);
        });
  }

  std::cout << "Modify stiffness mx" << std::endl;

  Eigen::SparseMatrix<double, Eigen::RowMajor> stiffnessMxMod =
      cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        domainCellGroup,
        NuTo::Visualize::VoronoiHandler(Visualize::VoronoiGeometryBrick(
            integrationOrder, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
    visualize.CellData(
        [&](const CellIpData cipd) {
          EngineeringStress<3> stress =
              steel.Stress(cipd.Apply(dof1, Nabla::Strain()), 0., cipd.Ids());
          return stress;
        },
        "stress");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  // ***********************************
  //    Solve
  // ***********************************

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

  // Set initial data
  Eigen::VectorXd w0(dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd v0(dofInfo.numIndependentDofs[dof1]);

  w0.setZero();
  v0.setZero();

  Eigen::VectorXd femResult(numDofs);

  auto MergeResult = [&](Eigen::VectorXd &v) {
    femResult = cmat * v + constraints.GetSparseGlobalRhs(dof1, numDofs, 0.);
    for (auto &node : mesh.NodesTotal(dof1)) {
      for (int component = 0; component < dof1.GetNum(); component++) {
        int dofNr = node.GetDofNumber(component);
        node.SetValue(component, femResult[dofNr]);
      }
    };
  };

  auto state = std::make_pair(w0, v0);

  Eigen::VectorXd fmod = -stiffnessMx(dof1, dof1) *
                         constraints.GetSparseGlobalRhs(dof1, numDofs, 0.0);

  Eigen::VectorXd fTime(numDofs);
  fTime.setZero();

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
    fTime[forceNodeId] = smearedStepFunction(t, riseTime);
    d2wdt2 = -stiffnessMxMod * w + cmat.transpose() * (fmod + fTime);
    d2wdt2.cwiseQuotient(massMxMod);
  };

  // ***********************
  // Solve
  // ***********************

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * timeStep;
    state = ti.DoStep(eq, state.first, state.second, t, timeStep);
    std::cout << i + 1 << std::endl;
    // plot
    if ((i * 100) % numSteps == 0) {
      MergeResult(state.first);
      visualizeResult(resultDirectoryFull.string() + "PointForce3D_" +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
