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

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include <iostream>

#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/MeshValuesTools.h"

using namespace NuTo;

/*  Solves Poisson equation on a Rectangle:
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
 * The definition of the normals in the 1D in 2D case is problematic
 * (to the right means when going in positive direction, i.e. counter clock
 * the normals point inwards. Added a minus in neumann load to acount for that)
 *
 * */
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

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](Eigen::Vector2d r) {
    return sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
  };
  auto solutionGradient = [](Eigen::Vector2d r) {
    Eigen::Vector2d result;
    result[0] = 2 * M_PI * cos(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    result[1] = 2 * M_PI * sin(2 * M_PI * r[0]) * cos(2 * M_PI * r[1]);
    return result;
  };
  auto negativeLaplaceOfSolution = [](Eigen::Vector2d r) {
    double result =
        8 * M_PI * M_PI * sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    return result;
  };

  Tools::SetValues(domain, dof2, solution);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

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
  CellStorage volumeCells;
  Group<CellInterface> volumeCellGroup =
      volumeCells.AddCells(domain, integrationType2D);

  // boundary cells
  CellStorage neumannBoundaryCells;
  Group<CellInterface> neumannBoundaryCellGroup =
      neumannBoundaryCells.AddCells(neumannBoundary, integrationType1D);

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  auto stiffnessF = [&](const CellIpData &cipd) {

    Eigen::MatrixXd B = cipd.B(dof1, Nabla::Gradient());
    DofMatrix<double> stiffnessLocal;
    stiffnessLocal(dof1, dof1) = B.transpose() * B;
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

  auto neumannloadF = [&](const CellIpData &cipd) {

    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;
    Eigen::VectorXd coords = cipd.GlobalCoordinates();
    Eigen::Vector2d f = solutionGradient(coords);
    Eigen::Vector2d normal = cipd.GetJacobian().Normal();

    double normalComponent = f.dot(normal);

    loadLocal[dof1] = N.transpose() * normalComponent;
    return loadLocal;
  };

  auto loadF = [&](const CellIpData &cipd) {

    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;
    Eigen::Vector2d coords = cipd.GlobalCoordinates();
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
  visualize.WriteVtuFile("PoissonRectangle.vtu");
}
