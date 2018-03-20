#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTriangle.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  // ***************************
  //      Import a mesh
  // ***************************

  MeshGmsh gmsh("circle.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto boundary = gmsh.GetPhysicalGroup("Boundary");
  auto dirichletBoundary = boundary;

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](Eigen::Vector2d r) {
    return exp(M_PI * r[0]) * sin(M_PI * r[1]);
  };

  // ***************************
  //      Add DoFs
  // ***************************

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1);

  DofType dof2("Exact", 1);
  AddDofInterpolation(&mesh, dof2);

  for (NuTo::ElementCollectionFem &elmColl : domain) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof2);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      Eigen::Vector2d coord =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      elmDof.GetNode(i).SetValue(0, solution(coord));
    }
  }

  NuTo::Integrands::PoissonTypeProblem<2> pde(dof1);

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

  IntegrationTypeTriangle integrationType2D(2);

  CellStorage cells;
  auto volumeCells = cells.AddCells(domain, integrationType2D);

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  GlobalDofMatrixSparse stiffnessMx = asmbl.BuildMatrix(
      volumeCells, {dof1},
      [&](const CellIpData &cipd) { return pde.StiffnessMatrix(cipd); });

  GlobalDofVector loadVector = asmbl.BuildVector(
      {}, {dof1}, [&](const CellIpData &cipd) { return pde.LoadVector(cipd); });

  // Compute modified stiffness matrix
  auto kJJ = stiffnessMx.JJ(dof1, dof1);
  auto kJK = stiffnessMx.JK(dof1, dof1);
  auto kKJ = stiffnessMx.KJ(dof1, dof1);
  auto kKK = stiffnessMx.KK(dof1, dof1);

  Eigen::SparseMatrix<double> stiffnessMxMod =
      kJJ - cmat.transpose() * kKJ - kJK * cmat + cmat.transpose() * kKK * cmat;

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

  NuTo::Visualize::Visualizer visualize(volumeCells,
                                        NuTo::Visualize::AverageHandler());
  visualize.DofValues(dof1);
  visualize.DofValues(dof2);
  visualize.WriteVtuFile("LaplaceCircle.vtu");
}
