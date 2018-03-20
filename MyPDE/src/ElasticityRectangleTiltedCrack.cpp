#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constitutive/EngineeringStrain.h"
#include "nuto/mechanics/constitutive/LinearElastic.h"

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

std::tuple<double, double, double>
CalculateCoefficients2DPlaneStress(double E, double Nu) {
  double factor = E / (1.0 - (Nu * Nu));
  return std::make_tuple(factor,                     // C11
                         factor * Nu,                // C12
                         factor * 0.5 * (1.0 - Nu)); // C33
}

std::tuple<double, double, double> CalculateCoefficients3D(double E,
                                                           double Nu) {
  double factor = E / ((1.0 + Nu) * (1.0 - 2.0 * Nu));
  return std::make_tuple(factor * (1.0 - Nu),    // C11
                         factor * Nu,            // C12
                         E / (2. * (1.0 + Nu))); // C33
}

Eigen::Matrix3d CalculateC(double E, double Nu, ePlaneState planeState) {
  double C11 = 0, C12 = 0, C33 = 0;
  if (planeState == ePlaneState::PLANE_STRESS)
    std::tie(C11, C12, C33) = CalculateCoefficients2DPlaneStress(E, Nu);
  else
    std::tie(C11, C12, C33) = CalculateCoefficients3D(E, Nu);

  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  C(0, 0) = C11;
  C(1, 0) = C12;

  C(0, 1) = C12;
  C(1, 1) = C11;

  C(2, 2) = C33;
  return C;
}

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

  Eigen::Vector2d crackLoad(1., 0.);

  double tau = 0.2e-6;
  double stepSize = 0.02e-6;
  int numSteps = 50000;

  // ***************************
  //      Import a mesh
  // ***************************

  MeshGmsh gmsh("rectangleWithInternalCrack01.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto frontCrackFace = gmsh.GetPhysicalGroup("LeftCrackFace");
  auto backCrackFace = gmsh.GetPhysicalGroup("RightCrackFace");
  auto crackBoundary = Unite(frontCrackFace, backCrackFace);

  DofType dof1("Displacements", 2);

  int order = 2;

  auto &ipol2D = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol2D);

  auto &ipol1D = mesh.CreateInterpolation(InterpolationTrussLobatto(order));
  AddDofInterpolation(&mesh, dof1, crackBoundary, ipol1D);

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
  IntegrationTypeTensorProduct<1> integrationType1D(
      order + 1, eIntegrationMethod::LOBATTO);

  // volume cells
  boost::ptr_vector<CellInterface> volumeCells;
  int cellId = 0;
  for (ElementCollection &element : domain) {
    volumeCells.push_back(new Cell(element, integrationType2D, cellId++));
  }
  Group<CellInterface> volumeCellGroup;
  for (CellInterface &c : volumeCells) {
    volumeCellGroup.Add(c);
  }

  // boundary cells
  boost::ptr_vector<CellInterface> neumannBoundaryCells;
  for (ElementCollection &element : crackBoundary) {
    neumannBoundaryCells.push_back(
        new Cell(element, integrationType1D, cellId++));
  }
  Group<CellInterface> neumannBoundaryCellGroup;
  for (CellInterface &c : neumannBoundaryCells) {
    neumannBoundaryCellGroup.Add(c);
  }

  // ***********************************
  //    Calculate system matrices
  // ***********************************

  using namespace std::placeholders;

  Eigen::MatrixXd stiffnessTensor =
      CalculateC(E, nu, ePlaneState::PLANE_STRAIN);

  auto stiffnessMxF = [&](const CellIpData &cipd) {
    BMatrixStrain B = cipd.B(dof1, Nabla::Strain());
    DofMatrix<double> hessian0;

    hessian0(dof1, dof1) = B.transpose() * stiffnessTensor * B;
    return hessian0;
  };

  auto massMxF = [&](const CellIpData &cipd) {

    Eigen::MatrixXd N = cipd.N(dof1);
    DofMatrix<double> massLocal;
    massLocal(dof1, dof1) = N.transpose() * rho * N;
    return massLocal;
  };

  auto boundaryLoadF = [&](const CellIpData &cipd) {
    Eigen::MatrixXd N = cipd.N(dof1);
    NuTo::DofVector<double> load;

    load[dof1] = N.transpose() * crackLoad;
    return load;
  };

  std::cout << "NumDofs: " << mesh.NodesTotal(dof1).Size() << std::endl;

  std::cout << "Calculate stiffness" << std::endl;
  auto stiffnessMx = asmbl.BuildMatrix(volumeCellGroup, {dof1}, stiffnessMxF);
  std::cout << "Calculate mass" << std::endl;
  auto massMx =
      asmbl.BuildDiagonallyLumpedMatrix(volumeCellGroup, {dof1}, massMxF);
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
      NuTo::Visualize::VoronoiHandler(
          NuTo::Visualize::VoronoiGeometryQuad(order + 1, Visualize::LOBATTO)));

  int plotcounter = 1;

  std::cout << "Start time integration" << std::endl;

  for (int i = 0; i < numSteps; i++) {
    std::cout << i * 100. / numSteps << std::endl;
    double t = i * stepSize;
    state = ti.DoStep(problemToSolve, state.first, state.second, t, stepSize);
    femResult.head(numDofsJ) = state.first;
    femVelocities.head(numDofsJ) = state.second;
    if ((i * 100) % numSteps == 0) {
      //      std::cout << plotcounter;
      MergeNodeVals(femResult, allNodes);
      visualize.DofValues(dof1);
      visualize.CellData(
          [&](const CellIpData &cipd) {
            EngineeringStrain<2> strain = cipd.Apply(dof1, Nabla::Strain());
            return strain;
          },
          "strain");
      visualize.WriteVtuFile("TiltedCrackRectangle_" +
                             std::to_string(plotcounter) + ".vtu");
      plotcounter++;
    }
  }
}
