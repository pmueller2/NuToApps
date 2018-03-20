#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/MeshValuesTools.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

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
  auto dirichletBoundary = Unite(top, bottom);
  auto neumannBoundary = Unite(left, right);

  // ***************************
  //      Add DoFs
  // ***************************

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1);

  DofType dof2("Exact", 1);
  AddDofInterpolation(&mesh, dof2);

  // Get Coordinates of nodes
  NuTo::Integrands::PoissonTypeProblem<2> pde(dof1);

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](Eigen::Vector2d r) { return exp(r[0]) * sin(r[1]); };
  auto solutionGradient = [](Eigen::Vector2d r) {
    Eigen::Vector2d result(sin(r[1]), cos(r[1]));
    result *= exp(r[0]);
    return result;
  };

  Tools::SetValues(domain, dof2, solution);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  // Apply boundary condition
  Constraint::Constraints constraints;
  constraints.Add(dof1, Constraint::SetDirichletBoundaryNodes(
                            dof1, dirichletBoundary, solution));

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
      3, eIntegrationMethod::LOBATTO);
  IntegrationTypeTensorProduct<2> integrationType2D(
      3, eIntegrationMethod::LOBATTO);

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
  //    Assemble load vector
  // ***********************************

  GlobalDofVector loadVector =
      asmbl.BuildVector(neumannBoundaryCellGroup, {dof1},
                        [&](const CellIpData &cipd) {
                          return pde.NormalNeumannLoad(cipd, solutionGradient);
                        });

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

  // ***********************************
  //    Visualize
  // ***********************************

  NuTo::Visualize::Visualizer visualize(volumeCellGroup,
                                        NuTo::Visualize::AverageHandler());
  visualize.DofValues(dof1);
  visualize.DofValues(dof2);
  visualize.WriteVtuFile("LaplaceRectangle.vtu");
}
