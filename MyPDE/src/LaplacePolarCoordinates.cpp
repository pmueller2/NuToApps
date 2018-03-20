#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include <iostream>

namespace NuTo {
namespace Nabla {
struct CylindricalGradient : Interface {
  CylindricalGradient(Eigen::Vector2d coords) {
    double r = coords[0];
    double phi = coords[1];
    Jinv << cos(phi), sin(phi), -sin(phi) / r, cos(phi) / r;
  }

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &dNdX) const override {
    return (dNdX * Jinv).transpose();
  }

private:
  Eigen::Matrix2d Jinv;
};
} // Nabla
} // NuTo

using namespace NuTo;

int main(int argc, char *argv[]) {

  // ***************************
  //   Generate coordinate mesh
  // ***************************

  int numR = 10;
  int numPhi = 36;

  double r0 = 0.1;
  double r1 = 1.0;

  MeshFem mesh1 = UnitMeshFem::CreateQuads(numR, numPhi);

  MeshFem mesh =
      UnitMeshFem::Transform(std::move(mesh1), [&](Eigen::Vector2d c) {
        return Eigen::Vector2d(r0 + (r1 - r0) * c[0], 2 * M_PI * c[1]);
      });

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](Eigen::Vector2d coords) {
    double r = coords[0];
    double phi = coords[1];
    double x = r * cos(phi);
    double y = r * sin(phi);
    return exp(M_PI * x) * sin(M_PI * y);
  };

  // ***************************
  //      Add DoFs
  // ***************************

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1);

  DofType dof2("Exact", 1);
  AddDofInterpolation(&mesh, dof2);

  for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof2);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      Eigen::Vector2d coord =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      elmDof.GetNode(i).SetValue(0, solution(coord));
    }
  }

  // Set up periodic boundary condition for phi
  // Get Coordinates of nodes
  std::map<NodeSimple *, Eigen::Vector2d> nodeCoordinateMap;
  for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      nodeCoordinateMap[&(elmDof.GetNode(i))] =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
    }
  }
  Constraint::Constraints constraints;

  Group<NodeSimple> nodesPhi0 = mesh.NodesAtAxis(eDirection::Y, dof1, 0);
  Group<NodeSimple> nodesPhi2Pi =
      mesh.NodesAtAxis(eDirection::Y, dof1, 2 * M_PI, 1e-6);

  for (auto &nd1 : nodesPhi0)
    for (auto &nd2 : nodesPhi2Pi) {
      double r_nd1 = nodeCoordinateMap[&nd1][0];
      if ((std::abs(r_nd1 - r0) < 1e-6) || (std::abs(r_nd1 - r1) < 1e-6))
        continue;
      double r_nd2 = nodeCoordinateMap[&nd2][0];
      if (std::abs(r_nd1 - r_nd2) < 1e-6) {
        Constraint::Term term2(nd1, 0, -1);
        Constraint::Equation constraintEq(nd2, 0, [](double) { return 0.; });
        constraintEq.AddTerm(term2);
        constraints.Add(dof1, constraintEq);
        continue;
      }
    }

  for (auto &nd : mesh.NodesTotal(dof1)) {
    double r = nodeCoordinateMap[&nd][0];
    if ((std::abs(r - r0) < 1e-6) || (std::abs(r - r1) < 1e-6))
      constraints.Add(dof1,
                      Constraint::Component(nd, {eDirection::X},
                                            solution(nodeCoordinateMap[&nd])));
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

  IntegrationTypeTensorProduct<2> integrationType2D(3,
                                                    eIntegrationMethod::GAUSS);

  CellStorage cells;
  auto volumeCells = cells.AddCells(mesh.ElementsTotal(), integrationType2D);

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  auto StiffnessMatrixF = [dof1](const CellIpData &cipd) {
    Eigen::MatrixXd B =
        cipd.B(dof1, Nabla::CylindricalGradient(cipd.GlobalCoordinates()));
    DofMatrix<double> stiffnessLocal;
    double CylindricalDeterminant = cipd.GlobalCoordinates()[0];
    stiffnessLocal(dof1, dof1) = CylindricalDeterminant * B.transpose() * B;
    return stiffnessLocal;
  };

  auto LoadVectorF = [dof1](const CellIpData &cipd) {
    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;
    loadLocal[dof1] = N.transpose() * 0.;
    return loadLocal;
  };

  GlobalDofMatrixSparse stiffnessMx =
      asmbl.BuildMatrix(volumeCells, {dof1}, StiffnessMatrixF);

  GlobalDofVector loadVector = asmbl.BuildVector({}, {dof1}, LoadVectorF);

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
  visualize.WriteVtuFile("LaplacePolarCoordinates.vtu");

  mesh = UnitMeshFem::Transform(std::move(mesh), [&](Eigen::Vector2d c) {
    double r = c[0];
    double phi = c[1];
    double x = r * cos(phi);
    double y = r * sin(phi);
    return Eigen::Vector2d(x, y);
  });

  NuTo::Visualize::Visualizer visualize2(volumeCells,
                                         NuTo::Visualize::AverageHandler());
  visualize2.DofValues(dof1);
  visualize2.DofValues(dof2);
  visualize2.WriteVtuFile("LaplacePolarPlot.vtu");
}
