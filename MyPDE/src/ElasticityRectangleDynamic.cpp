#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/constitutive/LinearElastic.h"

#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/tools/CellStorage.h"

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

#include <iostream>

using namespace NuTo;

/*
 * Solves dynamic Lame equation (isotropic homogeneous linear elasticity) on a
 * Rectangle:
 * */
int main(int argc, char *argv[]) {

  // **************************************
  //      Set some problem parameters
  // **************************************

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  double tau = 0.2e-6;
  double stepSize = 0.002e-6;
  int numSteps = 2;

  int order = 4;

  const Eigen::Vector2d neumannData(1000., 0.);
  auto pointLoadFunction = [tau](double t) {
    double ot = M_PI * t / tau;
    if (ot > M_PI)
      return 1.;
    if (ot < 0.)
      return 0.;
    return 0.5 * (1. - cos(ot));
  };

  // ***************************
  //      Import a mesh
  // ***************************

  MeshGmsh gmsh("rectangle10x10.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  double lX = 1.00;
  double lY = 1.00;

  double notch = 0.2;

  auto boundary = Unite(Unite(top, bottom), Unite(left, right));
  auto dirichletBoundary = left;
  auto neumannBoundary = right;

  DofType dof1("Displacements", 2);

  Laws::LinearElastic<2> steel(E, nu);
  Integrands::DynamicMomentumBalance<2> pde(dof1, steel, rho);

  auto &ipol = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol);

  auto &ipolBoundary =
      mesh.CreateInterpolation(InterpolationTrussLobatto(order));
  AddDofInterpolation(&mesh, dof1, boundary, ipolBoundary);

  // Get Coordinates of boundary nodes
  std::map<NodeSimple *, Eigen::Vector2d> nodeCoordinateMap;
  for (NuTo::ElementCollectionFem &elmColl : domain) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof1 = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof1.Interpolation().GetNumNodes(); i++) {
      nodeCoordinateMap[&(elmDof1.GetNode(i))] =
          Interpolate(elmCoord, elmDof1.Interpolation().GetLocalCoords(i));
    }
  }

  NodeSimple &rightBottomNode =
      mesh.NodeAtCoordinate(Eigen::Vector2d(lX, 0.0), dof1, 1.e-7);
  NodeSimple &leftBottomNode =
      mesh.NodeAtCoordinate(Eigen::Vector2d(0., 0.0), dof1, 1.e-7);
  NodeSimple &loadNode =
      mesh.NodeAtCoordinate(Eigen::Vector2d(0., lY), dof1, 1.e-7);

  Group<NodeSimple> dirichletBoundaryNodes;
  for (ElementCollectionFem &elmColl : dirichletBoundary) {
    NuTo::ElementFem &elmDof1 = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof1.GetNumNodes(); i++) {
      dirichletBoundaryNodes.Add(elmColl.DofElement(dof1).GetNode(i));
    }
  }

  // Set x component to 0 (symmetry boundary, except at notch)
  Constraint::Constraints constraints;
  for (auto &nd : dirichletBoundaryNodes) {
    double y = nodeCoordinateMap.at(&nd)[1];
    if (y <= (lY - notch))
      constraints.Add(dof1, Constraint::Component(nd, {eDirection::X}, 0.));
  }

  // Set y component to 0 at right bottom node
  constraints.Add(dof1,
                  Constraint::Component(rightBottomNode, {eDirection::Y}, 0.));

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
      order + 1, eIntegrationMethod::LOBATTO);
  IntegrationTypeTensorProduct<2> integrationType2D(
      order + 1, eIntegrationMethod::LOBATTO);

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

  auto neumannloadF = [&](const CellIpData &cipd, double /* t */) {

    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;
    Eigen::Vector2d f = neumannData;
    loadLocal[dof1] = N.transpose() * neumannData;
    return loadLocal;
  };

  // ***********************************
  //    ComputeLumpedMassMatrix
  // ***********************************

  GlobalDofVector lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      volumeCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  auto mJJ = lumpedMassMx.J[dof1];
  auto mKK = lumpedMassMx.K[dof1];
  Eigen::MatrixXd tmp = cmat.transpose() * mKK.asDiagonal() * cmat;
  DofVector<double> lumpedMassMxMod;
  lumpedMassMxMod[dof1] = mJJ + tmp.diagonal();

  // ***********************************
  //    Add boundary loads
  // ***********************************

  auto computeModifiedLoadVector = [&](double t3) {
    auto neumann2 = [&neumannloadF, t3](const CellIpData &cipd) {
      return neumannloadF(cipd, t3);
    };
    GlobalDofVector tmpLoad =
        asmbl.BuildVector(neumannBoundaryCellGroup, {dof1}, neumann2);
    // tmpLoad.J[dof1][loadNode.GetDofNumber(1)] -= pointLoadFunction(t3);
    // --------------------------
    double dt = 1e-14;
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
  //    Visualize
  // ***********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        volumeCellGroup,
        NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryQuad(
            order + 1, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
    visualize.CellData(
        [&](const CellIpData &cipd) {
          return cipd.Apply(dof1, Nabla::Strain());
        },
        "strain");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  auto MergeResult = [&mesh, &dof1](Eigen::VectorXd femResult) {
    for (auto &node : mesh.NodesTotal(dof1)) {
      for (int component = 0; component < dof1.GetNum(); component++) {
        int dofNr = node.GetDofNumber(component);
        node.SetValue(component, femResult[dofNr]);
      }
    };
  };
  // ***********************************
  //    Solve Static
  // ***********************************
  Eigen::VectorXd femResult(dofInfo.numIndependentDofs[dof1] +
                            dofInfo.numDependentDofs[dof1]);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(stiffnessMxMod);
  if (solver.info() != Eigen::Success)
    throw Exception("Decomposition failed");
  femResult.head(dofInfo.numIndependentDofs[dof1]) =
      solver.solve(computeModifiedLoadVector(0));
  if (solver.info() != Eigen::Success)
    throw Exception("Solve failed");

  femResult.tail(dofInfo.numDependentDofs[dof1]) =
      cmat * femResult.head(dofInfo.numIndependentDofs[dof1]) +
      constraints.GetRhs(dof1, 0.);

  MergeResult(femResult);

  visualizeResult("ElasticityRectangle_0");

  // ***********************************
  //    Solve
  // ***********************************

  // Set up second version of constraints, redo
  // numbering, and assembly
  // Then use fem Result node extract for initial conditions

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

  // Extract NodeVals
  for (auto &node : mesh.NodesTotal(dof1)) {
    int dofNr = node.GetDofNumber(0);
    femResult[dofNr] = node.GetValues()(0);
  }

  // Set initial data
  Eigen::VectorXd w0(dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd v0(dofInfo.numIndependentDofs[dof1]);

  w0.setZero();
  v0.setZero();

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
      visualizeResult("ElasticityRectangle_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
