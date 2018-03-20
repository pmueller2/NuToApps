#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  // ***************************
  //      Generate a mesh
  // ***************************

  int numElms = 5;
  MeshFem mesh = UnitMeshFem::CreateLines(numElms);
  double meshSize = 1. / numElms;

  // ***************************
  //      Add DoFs
  // ***************************

  const InterpolationSimple &interpolation1D =
      mesh.CreateInterpolation(InterpolationTrussLinear());

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1, interpolation1D);

  NuTo::Integrands::PoissonTypeProblem<1> pde(dof1);
  // ******************************
  //      Add boundary information
  // ******************************

  NodeSimple &nd0 =
      mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.), dof1);
  NodeSimple &nd1 =
      mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 1.), dof1);

  std::map<NodeSimple *, double> nodeCoordinateMap;
  nodeCoordinateMap[&nd0] = 0.;
  nodeCoordinateMap[&nd1] = 1.;
  std::map<NodeSimple *, double> nodeSurfaceNormalMap;
  nodeSurfaceNormalMap[&nd0] = -1;
  nodeSurfaceNormalMap[&nd1] = 1.;

  Group<NodeSimple> dirichletBoundary = {nd0};
  Group<NodeSimple> neumannBoundary = {nd1};

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](double r) { return (2.3 * r + 0.45); };
  auto solutionDerivative = [](double r) { return (2.3); };

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  Constraint::Constraints constraints;
  for (auto &nd : dirichletBoundary) {
    constraints.Add(dof1,
                    Constraint::Value(nd, solution(nodeCoordinateMap.at(&nd))));
  }

  // ******************************
  //    Set up assembler
  // ******************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildConstraintMatrix(dof1, dofInfo.numIndependentDofs[dof1]);
  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  IntegrationTypeTensorProduct<1> integrationType1D(
      5, eIntegrationMethod::LOBATTO);

  // volume cells
  boost::ptr_vector<CellInterface> volumeCells;
  int cellId = 0;
  for (ElementCollection &element : mesh.Elements) {
    volumeCells.push_back(new Cell(element, integrationType1D, cellId++));
  }
  Group<CellInterface> volumeCellGroup;
  for (CellInterface &c : volumeCells) {
    volumeCellGroup.Add(c);
  }

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  GlobalDofMatrixSparse stiffnessMx = asmbl.BuildMatrix(
      volumeCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.StiffnessMatrix(cipd); });

  // Compute modified stiffness matrix
  auto kJJ = stiffnessMx.JJ(dof1, dof1);
  auto kJK = stiffnessMx.JK(dof1, dof1);
  auto kKJ = stiffnessMx.KJ(dof1, dof1);
  auto kKK = stiffnessMx.KK(dof1, dof1);

  Eigen::SparseMatrix<double> stiffnessMxMod =
      kJJ - cmat.transpose() * kKJ - kJK * cmat + cmat.transpose() * kKK * cmat;

  // ***********************************
  //    Assemble load vector (zeros)
  // ***********************************

  GlobalDofVector loadVector = asmbl.BuildVector(
      volumeCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.LoadVector(cipd, 0.); });

  // ***********************************
  //    Add boundary loads
  // ***********************************

  for (auto &nd : neumannBoundary) {
    double normal = nodeSurfaceNormalMap.at(&nd);
    double coord = nodeCoordinateMap.at(&nd);
    loadVector.J[dof1](nd.GetDofNumber(0)) = normal * solutionDerivative(coord);
  }

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

  //  // ***********************************
  //  //    Visualize
  //  // ***********************************

  NuTo::Visualize::Visualizer visualize(
      volumeCellGroup,
      NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryLine(2)));
  visualize.DofValues(dof1);
  visualize.WriteVtuFile("LaplaceLine.vtu");
}
