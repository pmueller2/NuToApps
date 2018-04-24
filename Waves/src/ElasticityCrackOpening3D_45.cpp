#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

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

/* Rectangular Plate, linear isotropic homogeneous elasticity
 * Circular crack opening.
 */
int main(int argc, char *argv[]) {

  // *********************************
  //      Geometry parameter
  // *********************************

  MeshGmsh gmsh("Crack3D_angle45_h1_2ndOrder.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto frontCrackFace = gmsh.GetPhysicalGroup("BackCrackFace");
  auto backCrackFace = gmsh.GetPhysicalGroup("FrontCrackFace");
  auto crackBoundary = Unite(frontCrackFace, backCrackFace);

  // The normals produced by tethex are completely wrong
  // Check with Gmsh and the manually change signs

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 0.2e-6;
  double stepSize = 0.006e-6;
  int numSteps = 100000;

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  double crackAngle = 0.25 * M_PI;
  double crackRadius = 0.2;

  double crackArea = M_PI * crackRadius * crackRadius;
  double loadPressureMagnitude = 1. / crackArea;

  // Load on Front crack face
  Eigen::Vector3d crackLoad(0., -sin(crackAngle), cos(crackAngle));
  crackLoad *= loadPressureMagnitude;

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  int order = 2;

  auto &ipol3D = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol3D);

  auto &ipol2D = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, crackBoundary, ipol2D);

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
      crackBoundary.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage crackCellsFront;
  auto crackCellGroupFront =
      crackCellsFront.AddCells(frontCrackFace, *boundaryIntType);

  CellStorage crackCellsBack;
  auto crackCellGroupBack =
      crackCellsBack.AddCells(backCrackFace, *boundaryIntType);

  // *********************************************
  //      No Dirichlet boundary, Numbering
  // *********************************************

  Constraint::Constraints constraints;

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  int numDofs =
      dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

  std::cout << "NumDofs: " << numDofs << std::endl;

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  Eigen::VectorXd massMxMod = lumpedMassMx[dof1];

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  DofMatrixSparse<double> stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.Hessian0(cipd, 0.);
      });

  Eigen::SparseMatrix<double> stiffnessMxMod = stiffnessMx(dof1, dof1);

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        domainCellGroup,
        NuTo::Visualize::VoronoiHandler(Visualize::VoronoiGeometryBrick(
            integrationOrder, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
    visualize.CellData(
        [&](const CellIpData cipd) {
          EngineeringStress<3> stress =
              steel.Stress(cipd.Apply(dof1, Nabla::Strain()), 0., cipd.Ids());
          return stress;
        },
        "stress");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  // ***********************************
  //    Solve
  // ***********************************

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

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

    Eigen::Vector3d normalTraction = crackLoad * smearedStepFunction(tt, tau);

    loadLocal[dof1] = N.transpose() * normalTraction;
    return loadLocal;
  };

  // Compute load
  DofVector<double> boundaryLoadFront = asmbl.BuildVector(
      crackCellGroupFront, {dof1},
      [&](const CellIpData cipd) { return crackLoadFunc(cipd, tau * 2); });
  DofVector<double> boundaryLoadBack =
      asmbl.BuildVector(crackCellGroupBack, {dof1}, [&](const CellIpData cipd) {
        return crackLoadFunc(cipd, tau * 2);
      });
  Eigen::VectorXd loadVectorMod =
      boundaryLoadFront[dof1] - boundaryLoadBack[dof1];

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {

    Eigen::VectorXd tmp =
        (-stiffnessMxMod * w + loadVectorMod * smearedStepFunction(t, tau));
    d2wdt2 = (tmp.array() / massMxMod.array()).matrix();
  };

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * stepSize;
    state = ti.DoStep(eq, state.first, state.second, t, stepSize);
    std::cout << i + 1 << std::endl;
    if ((i * 100) % numSteps == 0) {
      MergeResult(state.first);
      visualizeResult("Crack3D_angle45_h1_NormalLoad2ndOrder" +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
