#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadQuadratic.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "../../MyTimeIntegration/RK4.h"
#include "../../NuToHelpers/MeshValuesTools.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include <set>

using namespace NuTo;

int main(int argc, char *argv[]) {

  // ***********************************
  //    Mesh
  // ***********************************

  MeshGmsh gmsh("rectangle100x100.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  auto dirichletBoundary = Intersection(top, bottom);
  auto neumannBoundary = bottom;

  // ***********************************
  //    Numerical paras
  // ***********************************

  double stepSize = 0.001;
  int numSteps = 2;
  int order = 1;

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  DofType dof1("Scalar", 1);

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
      neumannBoundary.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage neumannCells;
  auto neumannCellGroup =
      neumannCells.AddCells(neumannBoundary, *boundaryIntType);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  Constraint::Constraints constraints;
  if (!dirichletBoundary.Empty()) {
    throw Exception(__PRETTY_FUNCTION__, "Dirichlet boundary not implemented.");
  }

  // ******************************
  //      Set Initial Condition
  // ******************************

  // Dof Values
  auto initialData = [](Eigen::Vector2d coords) {
    double r = 0.2;              // Radius
    Eigen::Vector2d c(0.5, 0.5); // Center
    double d = (coords - c).norm();
    if (d < r) {
      return 0.5 * (1. + cos(M_PI * d / r));
    }
    return 0.;
  };

  Tools::SetValues(domain, dof1, initialData);

  // Velocities
  auto initialVelocities = [](Eigen::Vector2d coords) { return 0.; };

  Tools::SetValues(domain, dof1, initialVelocities, 1);

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  NuTo::Integrands::PoissonTypeProblem<2> pde(dof1);

  GlobalDofVector diagonalMass = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.MassMatrix(cipd); });

  // Here cmat stuff has to be added
  Eigen::VectorXd diagonalMassMod = diagonalMass.J[dof1];

  // ***********************************
  //    Assemble stiffness matrix (should not be needed - better compute the
  //    gradient directly)
  // ***********************************

  GlobalDofMatrixSparse stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.StiffnessMatrix(cipd);
      });

  // Here cmat stuff has to be added
  Eigen::SparseMatrix<double> stiffnessMxMod = stiffnessMx.JJ(dof1, dof1);

  // ************************************
  //   Set up equation system
  // ************************************

  // Set up state vector

  int numDofs =
      dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];
  Eigen::VectorXd values(numDofs);
  Eigen::VectorXd velocities(numDofs);

  // Extract NodeVals
  for (NodeSimple &node : mesh.NodesTotal(dof1)) {
    int dofNr = node.GetDofNumber(0);
    values[dofNr] = node.GetValues(0)(0);
    velocities[dofNr] = node.GetValues(0)(0);
  }

  // Concatenate independent values and velocities
  Eigen::VectorXd state(dofInfo.numIndependentDofs[dof1] * 2);
  state.head(dofInfo.numIndependentDofs[dof1]) =
      values.head(dofInfo.numIndependentDofs[dof1]);
  state.tail(dofInfo.numIndependentDofs[dof1]) =
      velocities.head(dofInfo.numIndependentDofs[dof1]);

  NuTo::TimeIntegration::RK4<Eigen::VectorXd> ti;

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &dwdt, double t) {
    // split into velocities and values
    Eigen::VectorXd valsJ = w.head(w.size() / 2);
    Eigen::VectorXd veloJ = w.tail(w.size() / 2);
    // compute dependent parts
    Eigen::VectorXd valsK(dofInfo.numDependentDofs[dof1]);
    Eigen::VectorXd veloK(dofInfo.numDependentDofs[dof1]);
    // combine
    Eigen::VectorXd vals(numDofs);
    vals << valsJ, valsK;
    Eigen::VectorXd velo(numDofs);
    velo << veloJ, veloK;
    // Merge
    for (auto &node : mesh.NodesTotal(dof1)) {
      int dofNr = node.GetDofNumber(0);
      node.SetValue(0, vals[dofNr], 0);
      node.SetValue(0, velo[dofNr], 1);
    }
    // **************************
    // Compute
    // **************************
    GlobalDofVector loadVector =
        asmbl.BuildVector(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.LoadVector(cipd,
                                [](Eigen::Vector2d coords) { return 0.; });
        });

    Eigen::VectorXd tmp = (-stiffnessMxMod * valsJ);
    tmp += loadVector.J[dof1];

    dwdt.head(w.size() / 2) = veloJ;
    dwdt.tail(w.size() / 2) = (tmp.array() / diagonalMassMod.array()).matrix();
  };

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(domainCellGroup,
                                          NuTo::Visualize::AverageHandler());
    visualize.DofValues(dof1);
    visualize.WriteVtuFile(filename + ".vtu");
  };

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    double t = i * stepSize;
    state = ti.DoStep(eq, state, t, stepSize);
    std::cout << i + 1 << std::endl;
    if ((i * 100) % numSteps == 0) {
      visualizeResult("Wave2Dlobatto1smooth_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
