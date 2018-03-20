#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"

#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include <functional>
#include <iostream>

using namespace NuTo;

void ExtractNodeVals(Eigen::VectorXd &femResult,
                     const Group<NodeSimple> &nodes) {
  for (NodeSimple &node : nodes) {
    for (int i = 0; i < node.GetNumValues(); i++) {
      int dofNr = node.GetDofNumber(i);
      femResult[dofNr] = node.GetValues()(i);
    }
  }
}

void MergeNodeVals(const Eigen::VectorXd &femResult, Group<NodeSimple> &nodes) {
  for (NodeSimple &node : nodes) {
    for (int i = 0; i < node.GetNumValues(); i++) {
      int dofNr = node.GetDofNumber(i);
      node.SetValue(i, femResult[dofNr]);
    }
  };
}

int main(int argc, char *argv[]) {

  // **************************************
  //      Set some problem parameters
  // **************************************

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  Eigen::Vector3d crackLoad(0., 0., 1.);

  double tau = 0.2e-6;
  double stepSize = 0.002e-6;
  int numSteps = 50000;

  // ***************************
  //      Import a mesh
  // ***************************

  MeshGmsh gmsh("plateWithInternalCrackHexed.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto frontCrackFace = gmsh.GetPhysicalGroup("FrontCrackFace");
  auto backCrackFace = gmsh.GetPhysicalGroup("BackCrackFace");
  auto crackBoundary = Unite(frontCrackFace, backCrackFace);

  auto neumannBoundary = Intersection(top, bottom);
  auto dirichletBoundary = Intersection(left, right);

  int order = 3;
  auto &ipol3D = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol3D);

  auto &ipol2D = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, crackBoundary, ipol2D);

  // ******************************
  //    Set up assembler
  // ******************************
  Constraint::Constraints constraints;

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildConstraintMatrix(dof1, dofInfo.numIndependentDofs[dof1]);

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  IntegrationTypeTensorProduct<2> integrationType2D(
      order + 1, eIntegrationMethod::LOBATTO);
  IntegrationTypeTensorProduct<3> integrationType3D(
      order + 1, eIntegrationMethod::LOBATTO);

  // volume cells
  CellStorage volumeCells;
  Group<CellInterface> volumeCellGroup =
      volumeCells.AddCells(domain, integrationType3D);

  // boundary cells
  CellStorage neumannBoundaryCells;
  Group<CellInterface> neumannBoundaryCellGroup =
      neumannBoundaryCells.AddCells(neumannBoundary, integrationType2D);

  // ***********************************
  //    Calculate system matrices
  // ***********************************

  auto boundaryLoadF = [&](const CellIpData &cipd) {
    Eigen::MatrixXd N = cipd.N(dof1);
    NuTo::DofVector<double> load;

    load[dof1] = N.transpose() * crackLoad;
    return load;
  };

  std::cout << "NumDofs: " << mesh.NodesTotal(dof1).Size() << std::endl;

  std::cout << "Calculate stiffness" << std::endl;
  auto stiffnessMx = asmbl.BuildMatrix(
      volumeCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian0(cipd, 0.); });

  std::cout << "Calculate mass" << std::endl;
  auto massMx = asmbl.BuildDiagonallyLumpedMatrix(
      volumeCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  std::cout << "Calculate load" << std::endl;
  auto loadVec =
      asmbl.BuildVector(neumannBoundaryCellGroup, {dof1}, boundaryLoadF);

  // Setup a solution vector
  int numDofsJ = dofInfo.numIndependentDofs[dof1];
  int numDofsK = dofInfo.numDependentDofs[dof1];
  int numDofs = numDofsJ + numDofsK;

  Eigen::VectorXd femResult(numDofs);
  femResult.setZero();
  Eigen::VectorXd femVelocities(numDofs);
  femVelocities.setZero();

  // TIME INTEGRATION

  auto problemToSolve = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2,
                            double t) {
    Eigen::VectorXd tmp = -stiffnessMx.JJ(dof1, dof1) * w + loadVec.J[dof1];
    d2wdt2 = (tmp.array() / massMx.J[dof1].array()).matrix();
  };

  auto allNodes = mesh.NodesTotal(dof1);

  ExtractNodeVals(femResult, allNodes);

  auto state =
      std::make_pair(femResult.head(numDofsJ), femVelocities.head(numDofsJ));

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  NuTo::Visualize::Visualizer visualize(
      volumeCellGroup,
      NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryBrick(
          order + 1, Visualize::LOBATTO)));

  int plotcounter = 1;

  std::cout << "Start time integration" << std::endl;

  for (int i = 0; i < numSteps; i++) {
    std::cout << i * 100. / numSteps << std::endl;
    double t = i * stepSize;
    state = ti.DoStep(problemToSolve, state.first, state.second, t, stepSize);
    femResult.head(numDofsJ) = state.first;
    femVelocities.head(numDofsJ) = state.second;
    if ((i * 100) % numSteps == 0) {
      std::cout << plotcounter;
      MergeNodeVals(femResult, allNodes);
      visualize.DofValues(dof1);
      visualize.CellData(
          [&](const CellIpData &cipd) {
            EngineeringStrain<3> strain = cipd.Apply(dof1, Nabla::Strain());
            return strain;
          },
          "strain");
      visualize.WriteVtuFile("TiltedCrack_" + std::to_string(plotcounter) +
                             ".vtu");
      plotcounter++;
    }
  }
}
