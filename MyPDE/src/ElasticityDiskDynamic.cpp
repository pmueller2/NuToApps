#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/constitutive/MechanicsInterface.h"

#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

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
 * Disk
 * */
int main(int argc, char *argv[]) {

  // **************************************
  //      Set some problem parameters
  // **************************************

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  double tau = 0.2e-6;
  double stepSize = 0.002e-6;
  int numSteps = 2000;

  Eigen::Vector3d loadCoordinate(0., 0., 0.01);

  int order = 5;

  const Eigen::Vector3d neumannData(0., 0., 0.);
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

  MeshGmsh gmsh("disk02.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto outerBoundary = gmsh.GetPhysicalGroup("OuterBoundary");
  auto bottomOuterRing = gmsh.GetPhysicalGroup("BottomOuterRing");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  auto boundary = Unite(Unite(top, bottom), outerBoundary);
  NuTo::Group<ElementCollectionFem> dirichletBoundary = bottomOuterRing;
  NuTo::Group<ElementCollectionFem> neumannBoundary;

  auto &ipol = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol);

  auto &ipolBoundary =
      mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, boundary, ipolBoundary);

  auto &ipolRing = mesh.CreateInterpolation(InterpolationTrussLobatto(order));
  AddDofInterpolation(&mesh, dof1, bottomOuterRing, ipolRing);

  NodeSimple &loadNode = mesh.NodeAtCoordinate(loadCoordinate, dof1, 1.e-6);

  Group<NodeSimple> dirichletBoundaryNodes;
  for (ElementCollectionFem &elmColl : dirichletBoundary) {
    NuTo::ElementFem &elmDof1 = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof1.GetNumNodes(); i++) {
      dirichletBoundaryNodes.Add(elmColl.DofElement(dof1).GetNode(i));
    }
  }

  Constraint::Constraints constraints;
  constraints.Add(
      dof1, Constraint::Component(dirichletBoundaryNodes, {eDirection::Z}, 0.));

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
  IntegrationTypeTensorProduct<3> integrationType3D(
      order + 1, eIntegrationMethod::LOBATTO);

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

  // volume cells
  boost::ptr_vector<CellInterface> neumannCells;
  cellId = 0;
  for (ElementCollection &element : neumannBoundary) {
    neumannCells.push_back(new Cell(element, integrationType2D, cellId++));
  }
  Group<CellInterface> neumannCellGroup;
  for (CellInterface &c : neumannCells) {
    neumannCellGroup.Add(c);
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

  auto neumannloadF = [&](const CellIpData &cipd, double /* t */) {

    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;
    Eigen::Vector3d f = neumannData;
    Eigen::Vector3d normal(0., 0., 0.);

    double normalComponent = f.dot(normal);

    loadLocal[dof1] = N.transpose() * normalComponent;
    return loadLocal;
  };

  // ***********************************
  //    ComputeLumpedMassMatrix
  // ***********************************

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      volumeCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  auto mJ = lumpedMassMx.J[dof1];
  auto mK = lumpedMassMx.K[dof1];

  Eigen::SparseMatrix<double> massModifier =
      cmat.transpose() * mK.asDiagonal() * cmat;

  Eigen::VectorXd massMxMod = mJ + massModifier.diagonal();

  // ***********************************
  //    Add boundary loads
  // ***********************************

  auto computeModifiedLoadVector = [&](double t3) {
    auto neumann2 = [&neumannloadF, t3](const CellIpData &cipd) {
      return neumannloadF(cipd, t3);
    };
    GlobalDofVector tmpLoad =
        asmbl.BuildVector(neumannCellGroup, {dof1}, neumann2);
    tmpLoad.J[dof1][loadNode.GetDofNumber(2)] = -pointLoadFunction(t3);
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
        cmat.transpose() * mK.asDiagonal() * bDDot;
    loadVectorMod += timeDepConstraintsPart;
    return loadVectorMod;
  };

  // ***********************************
  //    Visualize
  // ***********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        volumeCellGroup,
        NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryBrick(
            order + 1, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
    visualize.CellData(
        [&](const CellIpData &cipd) {
          return cipd.Apply(dof1, Nabla::Strain());
        },
        "strain");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  visualizeResult("ElasticityDisk_0");

  // ***********************************
  //    Solve
  // ***********************************

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

  Eigen::VectorXd femResult(dofInfo.numIndependentDofs[dof1] +
                            dofInfo.numDependentDofs[dof1]);
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

  auto MergeResult = [&mesh, &dof1](Eigen::VectorXd femResult) {
    for (auto &node : mesh.NodesTotal(dof1)) {
      for (int component = 0; component < dof1.GetNum(); component++) {
        int dofNr = node.GetDofNumber(component);
        node.SetValue(component, femResult[dofNr]);
      }
    };
  };

  auto state = std::make_pair(w0, v0);

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t1) {
    Eigen::VectorXd tmp = (-stiffnessMxMod * w + computeModifiedLoadVector(t1));
    d2wdt2 = (tmp.array() / massMxMod.array()).matrix();
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
      visualizeResult("ElasticityDisk_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
