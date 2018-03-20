#include "boost/ptr_container/ptr_vector.hpp"

#include "mechanics/interpolation/InterpolationQuadLinear.h"

#include "mechanics/mesh/MeshFemDofConvert.h"
#include "mechanics/mesh/MeshGmsh.h"

#include "mechanics/constraints/ConstraintCompanion.h"
#include "mechanics/constraints/Constraints.h"

#include "mechanics/cell/Cell.h"
#include "mechanics/cell/CellInterface.h"

#include "mechanics/cell/SimpleAssember.h"
#include "mechanics/dofs/DofNumbering.h"

#include "mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "../MyTimeIntegration/NY4NoVelocity.h"

#include "visualize/UnstructuredGrid.h"
#include "visualize/Visualizer.h"
#include "visualize/VoronoiGeometries.h"
#include "visualize/VoronoiHandler.h"

#include <iostream>

using namespace NuTo;

/*
 * Solves Wave equation on a Rectangle:
 *  d2u/dt2 - d2u/dx2 = f
 *
 * The corresponding FEM equation is then
 *
 * Mu + Ku = f + bndry
 *
 * Here K is the stiffness mx:    int (grad v)(grad u) dV
 *      M is the mass mx     :    int vu dV
 *      f load vector        :    int vf dV
 *      bndry boundary term  :    int vg dA
 *
 *
 * */
int main(int argc, char *argv[]) {

  // **************************************
  //      Set some problem parameters
  // **************************************

  double soundSpeed = 1.0;

  // ***************************
  //      Import a mesh
  // ***************************

  MeshGmsh gmsh("rectangle100x100.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  auto boundary = Unite(Unite(top, bottom), Unite(left, right));
  auto dirichletBoundary = Intersection(right, left);
  auto neumannBoundary = boundary;

  // ***************************
  //      Add DoFs
  // ***************************

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1);

  DofType dof2("Exact", 1);
  AddDofInterpolation(&mesh, dof2);

  // ******************************
  //    Add some node info
  // ******************************

  // Get Coordinates of nodes
  std::map<NodeSimple *, Eigen::Vector2d> nodeCoordinateMap;
  for (NuTo::ElementCollectionFem &elmColl : boundary) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof1 = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof1.Interpolation().GetNumNodes(); i++) {
      nodeCoordinateMap[&(elmDof1.GetNode(i))] =
          Interpolate(elmCoord, elmDof1.Interpolation().GetLocalCoords(i));
    }
  }

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [&soundSpeed](Eigen::Vector2d coords, double t) {
    double r = coords[0];
    if (((r - soundSpeed * t) > 0) && ((r - soundSpeed * t) < 1))
      return 0.5 * (1 - cos(2 * M_PI * (r - soundSpeed * t)));
    return 0.;
  };
  auto solutionSpaceDerivative = [&soundSpeed](Eigen::Vector2d coords,
                                               double t) {
    double r = coords[0];
    if (((r - soundSpeed * t) > 0) && ((r - soundSpeed * t) < 1))
      return Eigen::Vector2d(M_PI * sin(2 * M_PI * (r - soundSpeed * t)), 0.);
    return Eigen::Vector2d(0., 0.);
  };
  auto solutionTimeDerivative = [&soundSpeed](Eigen::Vector2d coords,
                                              double t) {
    double r = coords[0];
    if (((r - soundSpeed * t) > 0) && ((r - soundSpeed * t) < 1))
      return -soundSpeed * M_PI * sin(2 * M_PI * (r - soundSpeed * t));
    return 0.;
  };

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  // Create Group of all Dirichlet boundary nodes - no doubles
  Group<NodeSimple> dirichletBoundaryNodes;
  for (ElementCollectionFem &elmColl : dirichletBoundary) {
    for (int i = 0; i < elmColl.DofElement(dof1).GetNumNodes(); i++)
      dirichletBoundaryNodes.Add(elmColl.DofElement(dof1).GetNode(i));
  }

  Constraint::Constraints constraints;
  for (auto &nd : dirichletBoundaryNodes) {
    constraints.Add(dof1, Constraint::Value(nd, [&](double t) {
                      Eigen::Vector2d coord = nodeCoordinateMap.at(&nd);
                      return solution(coord, t);
                    }));
  }

  // ******************************
  //    Set up assembler
  // ******************************

  DofNumbering::DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildConstraintMatrix(dof1, dofInfo.numIndependentDofs[dof1]);
  SimpleAssembler asmbl =
      SimpleAssembler(dofInfo.numIndependentDofs, dofInfo.numDependentDofs);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  IntegrationTypeTensorProduct<1> integrationType1D(
      5, eIntegrationMethod::LOBATTO);
  IntegrationTypeTensorProduct<2> integrationType2D(
      5, eIntegrationMethod::LOBATTO);

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
  for (ElementCollection &element : neumannBoundary) {
    neumannBoundaryCells.push_back(
        new Cell(element, integrationType1D, cellId++));
  }
  Group<CellInterface> neumannBoundaryCellGroup;
  for (CellInterface &c : neumannBoundaryCells) {
    neumannBoundaryCellGroup.Add(c);
  }

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  auto stiffnessF = [&](const CellData /* &cellData */,
                        const CellIpData &cellIpData) {

    Eigen::MatrixXd B = cellIpData.GetBMatrixGradient(dof1);
    DofMatrix<double> stiffnessLocal;
    stiffnessLocal(dof1, dof1) = B.transpose() * B * soundSpeed * soundSpeed;
    return stiffnessLocal;
  };

  GlobalDofMatrixSparse stiffnessMx =
      asmbl.BuildMatrix(volumeCellGroup, {dof1}, stiffnessF);

  // Compute modified stiffness matrix
  auto kJJ = stiffnessMx.JJ(dof1, dof1);
  auto kJK = stiffnessMx.JK(dof1, dof1);
  auto kKJ = stiffnessMx.KJ(dof1, dof1);
  auto kKK = stiffnessMx.KK(dof1, dof1);

  Eigen::SparseMatrix<double> stiffnessMxMod =
      kJJ - cmat.transpose() * kKJ - kJK * cmat + cmat.transpose() * kKK * cmat;

  // ***********************************
  //    Assemble load vector
  // ***********************************

  auto neumannloadF = [&](const CellData /* &cellData */,
                          const CellIpData &cellIpData, double t) {

    Eigen::MatrixXd N = cellIpData.GetNMatrix(dof1);
    DofVector<double> loadLocal;
    Eigen::VectorXd coord = cellIpData.GlobalCoordinates();
    Eigen::Vector2d f =
        solutionSpaceDerivative(coord, t) * soundSpeed * soundSpeed;
    Eigen::Vector2d normal = -cellIpData.GetJacobian().Get().col(1);

    double normalComponent = f.dot(normal);

    loadLocal[dof1] = N.transpose() * normalComponent;
    return loadLocal;
  };

  // ***********************************
  //    ComputeLumpedMassMatrix
  // ***********************************

  GlobalDofVector lumpedMassMx;

  lumpedMassMx.J[dof1].resize(dofInfo.numIndependentDofs[dof1]);
  lumpedMassMx.K[dof1].resize(dofInfo.numDependentDofs[dof1]);

  lumpedMassMx.J[dof1].setZero();
  lumpedMassMx.K[dof1].setZero();

  auto massF = [&](const CellData /*&cellData*/, const CellIpData &cellIpData) {

    Eigen::MatrixXd N = cellIpData.GetNMatrix(dof1);
    DofMatrix<double> massLocal;

    massLocal(dof1, dof1) = N.transpose() * N;
    return massLocal;
  };

  auto totalMassFunction = [&](const CellData /* &cellData */,
                               const CellIpData /* &cellIpData */) {
    return 1;
  };

  for (NuTo::CellInterface &cell : volumeCellGroup) {

    // Compute local mass matrix
    DofMatrix<double> cellHessian = cell.Integrate(massF);
    // Now do mass lumping (sum all dimensions and later divide)
    const double totalMass = cell.Integrate(totalMassFunction);
    double diagonalMass = cellHessian(dof1, dof1).diagonal().sum();
    diagonalMass /= dof1.GetNum();
    double scaleFactor = totalMass / diagonalMass;
    // now scale all components
    Eigen::MatrixXd helper =
        (cellHessian(dof1, dof1).diagonal() * scaleFactor).asDiagonal();
    cellHessian(dof1, dof1) = helper;
    Eigen::VectorXi numberingdof = cell.DofNumbering(dof1);
    const Eigen::MatrixXd &cellHessianDof = cellHessian(dof1, dof1);

    for (int i = 0; i < numberingdof.rows(); ++i) {
      const int globalDofNumberI = numberingdof[i];
      const double globalDofValue = cellHessianDof(i, i);

      const bool activeI = globalDofNumberI < dofInfo.numIndependentDofs[dof1];

      if (activeI) {
        lumpedMassMx.J[dof1].coeffRef(globalDofNumberI) += globalDofValue;
      } else {
        lumpedMassMx.K[dof1].coeffRef(globalDofNumberI -
                                      dofInfo.numIndependentDofs[dof1]) +=
            globalDofValue;
      } // argh. any better ideas?
    }
  }

  auto mJJ = lumpedMassMx.J[dof1];
  auto mKK = lumpedMassMx.K[dof1];
  Eigen::MatrixXd tmp = cmat.transpose() * mKK.asDiagonal() * cmat;
  DofVector<double> lumpedMassMxMod;
  lumpedMassMxMod[dof1] = mJJ + tmp.diagonal();

  // ***********************************
  //    Add boundary loads
  // ***********************************

  auto computeModifiedLoadVector = [&](double t3) {
    auto neumann2 = [&neumannloadF, t3](const CellData &cellData,
                                        const CellIpData &cellIpData) {
      return neumannloadF(cellData, cellIpData, t3);
    };
    auto tmpLoad =
        asmbl.BuildVector(neumannBoundaryCellGroup, {dof1}, neumann2);
    // --------------------------
    double dt = 1e-5;
    // --------------------------
    auto fJ = tmpLoad.J[dof1];
    auto fK = tmpLoad.K[dof1];
    auto b = constraints.GetRhs(dof1, t3);
    auto bDDot = (constraints.GetRhs(dof1, t3) -
                  2. * constraints.GetRhs(dof1, t3 + 0.5 * dt) +
                  constraints.GetRhs(dof1, t3 + dt)) /
                 dt;
    Eigen::VectorXd loadVectorMod = fJ - cmat.transpose() * fK;
    loadVectorMod -= (kJK * b - cmat.transpose() * kKK * b);
    Eigen::VectorXd timeDepConstraintsPart =
        cmat.transpose() * mKK.asDiagonal() * bDDot;
    loadVectorMod += timeDepConstraintsPart;
    return loadVectorMod;
  };

  // ***********************************
  //    Set up initial condition
  // ***********************************

  Eigen::VectorXd femVelocities(dofInfo.numIndependentDofs[dof1] +
                                dofInfo.numDependentDofs[dof1]);

  // DOF 1 (initial condition)
  for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      Eigen::Vector2d coord =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      elmDof.GetNode(i).SetValue(0, solution(coord, 0.));
      femVelocities[elmDof.GetNode(i).GetDofNumber(0)] =
          solutionTimeDerivative(coord, 0);
    }
  }

  auto mergeExactSolution = [&](double t) {
    for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof2);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        Eigen::Vector2d coord =
            Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        double val = solution(coord, t);
        elmDof.GetNode(i).SetValue(0, val);
      }
    };
  };

  mergeExactSolution(0);

  // ***********************************
  //    Visualize
  // ***********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer<NuTo::Visualize::VoronoiHandler> visualize(
        volumeCellGroup, NuTo::Visualize::VoronoiGeometryQuad(2));
    visualize.DofValues(dof1);
    visualize.DofValues(dof2);
    visualize.WriteVtuFile(filename + ".vtu");
  };

  visualizeResult("WaveRectangle_0");

  // ***********************************
  //    Solve
  // ***********************************

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;
  double stepSize = 0.001;
  int numSteps = 1000;

  Eigen::VectorXd femResult(dofInfo.numIndependentDofs[dof1] +
                            dofInfo.numDependentDofs[dof1]);
  // Extract NodeVals
  for (auto &node : mesh.NodesTotal(dof1)) {
    int dofNr = node.GetDofNumber(0);
    femResult[dofNr] = node.GetValues()(0);
  }

  // Set initial data
  Eigen::VectorXd w0 = femResult.head(dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd v0 = femVelocities.head(dofInfo.numIndependentDofs[dof1]);

  auto MergeResult = [&mesh, &dof1](Eigen::VectorXd femResult) {
    for (auto &node : mesh.NodesTotal(dof1)) {
      int dofNr = node.GetDofNumber(0);
      node.SetValue(0, femResult[dofNr]);
    };
  };

  auto state = std::make_pair(w0, v0);

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t1) {
    Eigen::VectorXd tmp = (-stiffnessMxMod * w + computeModifiedLoadVector(t1));
    d2wdt2 = (tmp.array() / lumpedMassMx.J[dof1].array()).matrix();
  };

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * stepSize;
    state = ti.DoStep(eq, state.first, state.second, t, stepSize);
    femResult.head(dofInfo.numIndependentDofs[dof1]) = state.first;
    // Compute Dependent Dofs
    femResult.tail(dofInfo.numDependentDofs[dof1]) =
        -cmat * state.first + constraints.GetRhs(dof1, (i + 1) * stepSize);
    std::cout << i + 1 << std::endl;
    if ((i * 100) % numSteps == 0) {
      MergeResult(femResult);
      mergeExactSolution((i + 1) * stepSize);
      visualizeResult("WaveRectangle_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
