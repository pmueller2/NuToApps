#include "boost/ptr_container/ptr_vector.hpp"

#include "mechanics/mesh/MeshFemDofConvert.h"
#include "mechanics/mesh/MeshGmsh.h"

#include "mechanics/constraints/ConstraintCompanion.h"
#include "mechanics/constraints/Constraints.h"

#include "mechanics/cell/Cell.h"
#include "mechanics/cell/CellInterface.h"

#include "mechanics/cell/SimpleAssember.h"
#include "mechanics/dofs/DofNumbering.h"

#include "mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "visualize/UnstructuredGrid.h"
#include "visualize/Visualizer.h"
#include "visualize/VoronoiGeometries.h"
#include "visualize/VoronoiHandler.h"

#include <iostream>

using namespace NuTo;

/*  Solves Poisson equation on a Cube:
 *  - Delta u = f
 *
 * The corresponding FEM equation is then
 *
 * Ku = f + bndry
 *
 * Here K is the stiffness mx:    int (grad v)(grad u) dV
 *      f load vector        :    int vf dV
 *      bndry boundary term  :    int vg dA
 *
 *
 *
 * */
int main(int argc, char *argv[]) {

  // ***************************
  //      Import a mesh
  // ***************************

  std::cout << "Gmsh Import" << std::endl;

  MeshGmsh gmsh("cube20.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto front = gmsh.GetPhysicalGroup("Front");
  auto back = gmsh.GetPhysicalGroup("Back");
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Volume");

  std::cout << "Generate boundary" << std::endl;

  auto boundary =
      Unite(Unite(Unite(top, bottom), Unite(left, right)), Unite(front, back));
  //  auto dirichletBoundary = Unite(Unite(top, bottom), front);
  auto dirichletBoundary = boundary;
  auto neumannBoundary = Intersection(left, right);

  // ***************************
  //      Add DoFs
  // ***************************

  std::cout << "Add Dofs" << std::endl;

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1);

  DofType dof2("Exact", 1);
  AddDofInterpolation(&mesh, dof2);

  std::cout << "Create Coordinate Node Map" << std::endl;

  // Get Coordinates of nodes
  std::map<NodeSimple *, Eigen::Vector3d> nodeCoordinateMap;
  for (NuTo::ElementCollectionFem &elmColl : boundary) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      nodeCoordinateMap[&(elmDof.GetNode(i))] =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
    }
  }

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](Eigen::Vector3d r) {
    return sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]) * sin(2 * M_PI * r[2]);
  };
  auto solutionGradient = [](Eigen::Vector3d r) {
    Eigen::Vector3d result;
    result[0] = 2 * M_PI * cos(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]) *
                sin(2 * M_PI * r[2]);
    result[1] = 2 * M_PI * sin(2 * M_PI * r[0]) * cos(2 * M_PI * r[1]) *
                sin(2 * M_PI * r[2]);
    result[2] = 2 * M_PI * sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]) *
                cos(2 * M_PI * r[2]);
    return result;
  };
  auto negativeLaplaceOfSolution = [](Eigen::Vector3d r) {
    double result = 12 * M_PI * M_PI * sin(2 * M_PI * r[0]) *
                    sin(2 * M_PI * r[1]) * sin(2 * M_PI * r[2]);
    return result;
  };

  std::cout << "Set up exact solution" << std::endl;

  for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof2);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      Eigen::Vector3d coord =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      elmDof.GetNode(i).SetValue(0, solution(coord));
    }
  }

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  std::cout << "Set Dirichlet boundary" << std::endl;

  // Apply boundary condition
  Group<NodeSimple> dirichletBoundaryNodes;
  for (ElementCollectionFem &elmColl : dirichletBoundary) {
    for (int i = 0; i < elmColl.DofElement(dof1).GetNumNodes(); i++)
      dirichletBoundaryNodes.Add(elmColl.DofElement(dof1).GetNode(i));
  }

  Constraint::Constraints constraints;
  for (auto &nd : dirichletBoundaryNodes) {
    Eigen::Vector3d coord = nodeCoordinateMap.at(&nd);
    constraints.Add(dof1, Constraint::Value(nd, solution(coord)));
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

  IntegrationTypeTensorProduct<3> integrationType3D(
      3, eIntegrationMethod::LOBATTO);
  IntegrationTypeTensorProduct<2> integrationType2D(
      3, eIntegrationMethod::LOBATTO);

  std::cout << "Add cells" << std::endl;

  // volume cells
  boost::ptr_vector<CellInterface> volumeCells;
  int cellId = 0;
  for (ElementCollection &element : domain) {
    volumeCells.push_back(new Cell(element, integrationType3D, cellId++));
  }
  Group<CellInterface> volumeCellGroup;
  for (CellInterface &c : volumeCells) {
    volumeCellGroup.Add(c);
  }

  // boundary cells
  boost::ptr_vector<CellInterface> neumannBoundaryCells;
  for (ElementCollection &element : neumannBoundary) {
    neumannBoundaryCells.push_back(
        new Cell(element, integrationType2D, cellId++));
  }
  Group<CellInterface> neumannBoundaryCellGroup;
  for (CellInterface &c : neumannBoundaryCells) {
    neumannBoundaryCellGroup.Add(c);
  }

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  auto stiffnessF = [&](const CellData &cellData,
                        const CellIpData &cellIpData) {

    Eigen::MatrixXd B = cellIpData.GetBMatrixGradient(dof1);
    DofMatrix<double> stiffnessLocal;
    stiffnessLocal(dof1, dof1) = B.transpose() * B;
    return stiffnessLocal;
  };

  std::cout << "Assemble Stiffness" << std::endl;

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

  auto neumannloadF = [&](const CellData &cellData,
                          const CellIpData &cellIpData) {

    Eigen::MatrixXd N = cellIpData.GetNMatrix(dof1);
    DofVector<double> loadLocal;
    Eigen::VectorXd coords = cellIpData.GlobalCoordinates();
    Eigen::Vector3d f = solutionGradient(coords);
    Eigen::Vector3d normal = cellIpData.GetJacobian().Get().col(2);

    double normalComponent = f.dot(normal);

    loadLocal[dof1] = N.transpose() * normalComponent;
    return loadLocal;
  };

  std::cout << "Assemble Load" << std::endl;

  auto loadF = [&](const CellData &cellData, const CellIpData &cellIpData) {

    Eigen::MatrixXd N = cellIpData.GetNMatrix(dof1);
    DofVector<double> loadLocal;
    Eigen::Vector3d coords = cellIpData.GlobalCoordinates();
    double f = negativeLaplaceOfSolution(coords);
    loadLocal[dof1] = N.transpose() * f;
    return loadLocal;
  };

  GlobalDofVector neumannLoadVector =
      asmbl.BuildVector(neumannBoundaryCellGroup, {dof1}, neumannloadF);

  GlobalDofVector loadVector =
      asmbl.BuildVector(volumeCellGroup, {dof1}, loadF);

  // Add them
  loadVector += neumannLoadVector;

  // Compute modified load vector
  auto fJ = loadVector.J[dof1];
  auto fK = loadVector.K[dof1];
  auto b = constraints.GetRhs(dof1, 0);
  Eigen::VectorXd loadVectorMod = fJ - cmat.transpose() * fK;
  loadVectorMod -= (kJK * b - cmat.transpose() * kKK * b);

  // ***********************************
  //    Solve
  // ***********************************

  // Compute Independent Dofs
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

  std::cout << "Start solve" << std::endl;
  solver.compute(stiffnessMxMod);
  if (solver.info() != Eigen::Success) {
    throw Exception("decomposition failed");
  }
  Eigen::VectorXd result = solver.solve(loadVectorMod);
  if (solver.info() != Eigen::Success) {
    throw Exception("solve failed");
  }
  // Compute Dependent Dofs
  Eigen::VectorXd y = -cmat * result + b;

  // StoreResult
  Eigen::VectorXd femResult(result.size() + y.size());
  femResult.head(result.size()) = result;
  femResult.tail(y.size()) = y;

  // Merge
  for (auto &node : mesh.NodesTotal(dof1)) {
    int dofNr = node.GetDofNumber(0);
    node.SetValue(0, femResult[dofNr]);
  }

  // ***********************************
  //    Visualize
  // ***********************************

  NuTo::Visualize::Visualizer<NuTo::Visualize::VoronoiHandler> visualize(
      volumeCellGroup, NuTo::Visualize::VoronoiGeometryBrick(2));
  visualize.DofValues(dof1);
  visualize.DofValues(dof2);
  visualize.WriteVtuFile("PoissonCube.vtu");
}
