#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/MomentumBalance.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"

#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/mechanics/constitutive/MechanicsInterface.h"

#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  // **************************************
  //      Set some problem parameters
  // **************************************

  double E = 1.0;
  double nu = 0.3;
  double Iz = 1.;
  double P = 1.;
  double h = 0.5;
  double L = 20.0;

  // **************************************
  //      Set up solution
  // **************************************

  auto solution = [E, Iz, P, nu, h, L](Eigen::Vector2d coords) {
    double x = coords[0];
    double y = coords[1];
    //
    double ux = -P / (2 * E * Iz) * (L * L - x * x) * y -
                (P * (2 + nu)) / (6 * E * Iz) * y * y * y +
                (P * (1. + nu) * h * h) / (8 * E * Iz) * y;
    double uy = -(P * L * L * L) / (6 * E * Iz * Iz * Iz) *
                (2. - (3 * x) / L * (1. - (nu * y * y) / (L * L)) +
                 (x * x * x) / (L * L * L) +
                 (3 * h * h) / (4L * L) * (1 + nu) * (1. - x / L));

    return Eigen::Vector2d(ux, uy);
  };

  // ***************************
  //      Import a mesh
  // ***************************

  // MeshGmsh gmsh("LshapedDomainQuad0.msh");
  // MeshGmsh gmsh("LshapedDomainQuad1.msh");
  MeshGmsh gmsh("rectangleTest.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  //  auto cornerTop = gmsh.GetPhysicalGroup("cornerTop");
  //  auto cornerBottom = gmsh.GetPhysicalGroup("cornerBottom");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  //  NuTo::Group<ElementCollectionFem> boundary =
  //      Unite(Unite(Unite(cornerTop, cornerBottom), Unite(top, bottom)),
  //            Unite(left, right));

  NuTo::Group<ElementCollectionFem> boundary =
      Unite(Unite(top, bottom), Unite(left, right));
  NuTo::Group<ElementCollectionFem> dirichletBoundary = boundary;

  NuTo::Group<ElementCollectionFem> neumannBoundary;

  // ***************************
  //      Add DoFs
  // ***************************

  DofType dof1("Displacements", 2);
  DofType dof2("Exact", 2);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::MomentumBalance<3> pde(dof1, steel);

  int order = 1;
  AddDofInterpolation(&mesh, dof1);
  AddDofInterpolation(&mesh, dof2);

  // ******************************
  //    Add some node info
  // ******************************

  // Get Coordinates of boundary nodes
  std::map<NodeSimple *, Eigen::Vector2d> nodeCoordinateMap;
  for (NuTo::ElementCollectionFem &elmColl : domain) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof1 = elmColl.DofElement(dof1);
    NuTo::ElementFem &elmDof2 = elmColl.DofElement(dof2);
    for (int i = 0; i < elmDof1.Interpolation().GetNumNodes(); i++) {
      nodeCoordinateMap[&(elmDof1.GetNode(i))] =
          Interpolate(elmCoord, elmDof1.Interpolation().GetLocalCoords(i));
    }
    for (int i = 0; i < elmDof2.Interpolation().GetNumNodes(); i++) {
      nodeCoordinateMap[&(elmDof2.GetNode(i))] =
          Interpolate(elmCoord, elmDof2.Interpolation().GetLocalCoords(i));
    }
  }

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  // Create Group of all Dirichlet boundary dof nodes - no doubles
  Group<NodeSimple> dirichletBoundaryNodes;
  for (ElementCollectionFem &elmColl : dirichletBoundary) {
    for (int i = 0; i < elmColl.DofElement(dof1).GetNumNodes(); i++)
      dirichletBoundaryNodes.Add(elmColl.DofElement(dof1).GetNode(i));
  }
  Constraint::Constraints constraints;
  for (auto &nd : dirichletBoundaryNodes) {
    Eigen::Vector2d coords = nodeCoordinateMap.at(&nd);
    Eigen::Vector2d displ = solution(coords);
    constraints.Add(dof1, Constraint::Component(nd, {eDirection::X}, displ[0]));
    constraints.Add(dof1, Constraint::Component(nd, {eDirection::Y}, displ[1]));
  }

  // ******************************
  //    Merge exact solution
  // ******************************

  for (NodeSimple &nd : mesh.NodesTotal(dof2)) {
    Eigen::Vector2d coords = nodeCoordinateMap.at(&nd);
    Eigen::Vector2d displ = solution(coords);
    nd.SetValues(displ);
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

  IntegrationTypeTensorProduct<1> integrationType1D(order,
                                                    eIntegrationMethod::GAUSS);
  IntegrationTypeTensorProduct<2> integrationType2D(order,
                                                    eIntegrationMethod::GAUSS);

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
      [&](const CellIpData &cipd) { return pde.Hessian0(cipd, 0.); });

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

  auto loadF = [&](const CellIpData &cipd) {

    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;

    loadLocal[dof1] = N.transpose() * Eigen::Vector2d(0., 0.);
    return loadLocal;
  };

  GlobalDofVector loadVec = asmbl.BuildVector(volumeCellGroup, {dof1}, loadF);

  auto fJ = loadVec.J[dof1];
  auto fK = loadVec.K[dof1];
  auto b = constraints.GetRhs(dof1, 0.);
  Eigen::VectorXd loadVectorMod = fJ - cmat.transpose() * fK;
  loadVectorMod -= (kJK * b - cmat.transpose() * kKK * b);

  // ***********************************
  //    Add boundary loads
  // ***********************************

  // ***********************************
  //    Solve
  // ***********************************

  Eigen::VectorXd femResult(dofInfo.numIndependentDofs[dof1] +
                            dofInfo.numDependentDofs[dof1]);

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(stiffnessMxMod);
  if (solver.info() != Eigen::Success) {
    throw Exception("Decomposition failed");
  }
  Eigen::VectorXd solverResult = solver.solve(loadVectorMod);
  if (solver.info() != Eigen::Success) {
    throw Exception("Solvr failed");
  }
  femResult.head(dofInfo.numIndependentDofs[dof1]) = solverResult;
  // Compute Dependent Dofs
  femResult.tail(dofInfo.numDependentDofs[dof1]) =
      -cmat * solverResult + constraints.GetRhs(dof1, 0.);

  // ***********************************
  //    Merge
  // ***********************************

  auto MergeResult = [&mesh, &dof1](Eigen::VectorXd femResult) {
    for (auto &node : mesh.NodesTotal(dof1)) {
      for (int component = 0; component < dof1.GetNum(); component++) {
        int dofNr = node.GetDofNumber(component);
        node.SetValue(component, femResult[dofNr]);
      }
    };
  };

  MergeResult(femResult);

  // ***********************************
  //    Visualize
  // ***********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        volumeCellGroup,
        NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryQuad(
            order, NuTo::Visualize::GAUSS)));
    visualize.DofValues(dof1);
    visualize.DofValues(dof2);
    visualize.CellData(
        [&](const CellIpData &cipd) {
          return cipd.Apply(dof1, Nabla::Strain());
        },
        "strain");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  visualizeResult("ElasticityLShapedDomain");
}
