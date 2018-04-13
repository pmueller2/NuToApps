#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/Visualizer.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"
#include "../../NuToHelpers/ConstraintsHelper.h"

#include <iostream>

using namespace NuTo;

double smearedStepFunction(double t, double tau) {
  double ot = M_PI * t / tau;
  if (ot > M_PI)
    return 1.;
  if (ot < 0.)
    return 0.;
  return 0.5 * (1. - cos(ot));
}

/* Rectangular domain, linear isotropic homogeneous elasticity
 * Crack opening
 */
int main(int argc, char *argv[]) {

  // *********************************
  //      Geometry parameter
  // *********************************

  MeshGmsh gmsh("Crack2D_2.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto crack = gmsh.GetPhysicalGroup("Crack");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 0.2e-6;
  double stepSize = 0.006e-6;
  int numSteps = 30000;

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;
  Eigen::Vector2d crackLoad(-1., 0.);

  DofType dof1("Displacements", 2);

  Laws::LinearElastic<2> steel(E, nu); // 2D default: plane stress
  Integrands::DynamicMomentumBalance<2> pde(dof1, steel, rho);

  int order = 3;
  auto &ipol = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, ipol);
  mesh.AllocateDofInstances(dof1, 2);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  int integrationOrder = order + 1;

  // Domain cells
  auto domainIntType = CreateLobattoIntegrationType(
      domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage domainCells;
  auto domainCellGroup = domainCells.AddCells(domain, *domainIntType);

  // Boundary cells
  auto boundaryIntType = CreateLobattoIntegrationType(
      crack.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage crackCells;
  auto crackCellGroup = crackCells.AddCells(crack, *boundaryIntType);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  Group<NodeSimple> bottomNodes = GetNodes(bottom, dof1);
  Group<NodeSimple> leftNodes = GetNodes(left, dof1);

  Constraint::Constraints constraints;
  constraints.Add(dof1, Constraint::Component(bottomNodes, {eDirection::Y}));
  constraints.Add(dof1, Constraint::Component(leftNodes, {eDirection::X}));

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildConstraintMatrix(dof1, dofInfo.numIndependentDofs[dof1]);

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  auto mJ = lumpedMassMx.J[dof1];
  auto mK = lumpedMassMx.K[dof1];

  Eigen::SparseMatrix<double> massModifier =
      cmat.transpose() * mK.asDiagonal() * cmat;

  Eigen::VectorXd massMxMod = mJ + massModifier.diagonal();

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  GlobalDofMatrixSparse stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.Hessian0(cipd, 0.);
      });

  // Compute modified stiffness matrix
  auto kJJ = stiffnessMx.JJ(dof1, dof1);
  auto kJK = stiffnessMx.JK(dof1, dof1);
  auto kKJ = stiffnessMx.KJ(dof1, dof1);
  auto kKK = stiffnessMx.KK(dof1, dof1);

  Eigen::SparseMatrix<double> stiffnessMxMod =
      kJJ - cmat.transpose() * kKJ - kJK * cmat + cmat.transpose() * kKK * cmat;

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(domainCellGroup,
                                          NuTo::Visualize::AverageHandler());
    visualize.DofValues(dof1);
    visualize.WriteVtuFile(filename + ".vtu");
  };

  // ***********************************
  //    Solve
  // ***********************************

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

  Eigen::VectorXd femResult(dofInfo.numIndependentDofs[dof1] +
                            dofInfo.numDependentDofs[dof1]);

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

  auto crackLoadFunc = [&](const CellIpData &cipd, double tt) {
    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;

    Eigen::Vector2d normalTraction = crackLoad * smearedStepFunction(tt, tau);

    loadLocal[dof1] = N.transpose() * normalTraction;
    return loadLocal;
  };

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
    // Compute load
    GlobalDofVector boundaryLoad =
        asmbl.BuildVector(crackCellGroup, {dof1}, [&](const CellIpData cipd) {
          return crackLoadFunc(cipd, t);
        });
    // Include constraints
    auto fJ = boundaryLoad.J[dof1];
    auto fK = boundaryLoad.K[dof1];
    auto b = constraints.GetRhs(dof1, t);
    Eigen::VectorXd loadVectorMod = fJ - cmat.transpose() * fK;
    loadVectorMod -= (kJK * b - cmat.transpose() * kKK * b);

    Eigen::VectorXd tmp = (-stiffnessMxMod * w + loadVectorMod);
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
      visualizeResult("Crack2DNormalLoad_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
