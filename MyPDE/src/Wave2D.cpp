#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadQuadratic.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "../../MyTimeIntegration/RK4.h"
#include "../../NuToHelpers/ConstraintsHelper.h"
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

  auto dirichletBoundary = right;
  auto neumannBoundary = bottom;

  // ***********************************
  //    Numerical paras
  // ***********************************

  double stepSize = 0.001;
  double tau = 0.100;
  int numSteps = 1000;
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

  std::vector<Constraint::Equation> equations;
  std::set<NodeSimple *> nodes;

  for (NuTo::ElementCollectionFem &elmColl : dirichletBoundary) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      NodeSimple &nd = elmDof.GetNode(i);
      // If this node was already used: continue with next one
      if (nodes.find(&nd) != nodes.end())
        continue;
      nodes.insert(&nd);
      Eigen::VectorXd coords =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      equations.push_back(Constraint::Value(nd, [tau, coords](double t3) {
        double factor;
        if ((coords(1) <= 0.4) || (coords(1) >= 0.6)) {
          factor = 0.;
        } else {
          factor = 1.;
        }
        if (t3 >= tau) {
          return 0.;
        } else {
          return factor * 0.5 * (1. - cos(2 * M_PI * t3 / tau));
        }
      }));
    }
  }
  constraints.Add(dof1, equations);

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

  // Tools::SetValues(domain, dof1, initialData);

  // Velocities
  auto initialVelocities = [](Eigen::Vector2d coords) { return 0.; };

  Tools::SetValues(domain, dof1, initialVelocities, 1);

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  int numDofs =
      dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

  auto cmat = constraints.BuildUnitConstraintMatrix(dof1, numDofs);

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  NuTo::Integrands::PoissonTypeProblem<2> pde(dof1);

  DofVector<double> diagonalMass = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.MassMatrix(cipd); });

  // Here cmat stuff has to be added
  Eigen::VectorXd diagonalMassMod =
      (cmat.transpose() * diagonalMass[dof1].asDiagonal() * cmat)
          .eval()
          .diagonal();

  // ***********************************
  //    Assemble stiffness matrix (should not be needed - better compute the
  //    gradient directly)
  // ***********************************

  DofMatrixSparse<double> stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.StiffnessMatrix(cipd);
      });

  // Here cmat stuff has to be added
  Eigen::SparseMatrix<double> stiffnessMxMod =
      cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

  // ************************************
  //   Set up equation system
  // ************************************

  // Set up state vector

  Eigen::VectorXd values(numDofs);
  Eigen::VectorXd velocities(numDofs);

  // Extract NodeVals
  for (NodeSimple &node : mesh.NodesTotal(dof1)) {
    int dofNr = node.GetDofNumber(0);
    values[dofNr] = node.GetValues(0)(0);
    velocities[dofNr] = node.GetValues(0)(0);
  }

  // Get JK numbering
  Eigen::VectorXi jknumbering =
      Constraint::GetJKNumbering(constraints, dof1, numDofs);
  Eigen::PermutationMatrix<Eigen::Dynamic> P(jknumbering);

  // Reorder values
  Eigen::VectorXd orderedvalues = P.transpose() * values;
  Eigen::VectorXd orderedvelocities = P.transpose() * velocities;

  // Concatenate independent values and velocities
  Eigen::VectorXd state(dofInfo.numIndependentDofs[dof1] * 2);
  state.head(dofInfo.numIndependentDofs[dof1]) =
      orderedvalues.head(dofInfo.numIndependentDofs[dof1]);
  state.tail(dofInfo.numIndependentDofs[dof1]) =
      orderedvelocities.head(dofInfo.numIndependentDofs[dof1]);

  NuTo::TimeIntegration::RK4<Eigen::VectorXd> ti;

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &dwdt, double t) {
    // split into velocities and values
    Eigen::VectorXd valsJ = w.head(w.size() / 2);
    Eigen::VectorXd veloJ = w.tail(w.size() / 2);
    // compute dependent parts
    Eigen::VectorXd vals =
        cmat * valsJ + constraints.GetSparseGlobalRhs(dof1, numDofs, t);
    Eigen::VectorXd velo = cmat * veloJ;
    // Merge
    for (auto &node : mesh.NodesTotal(dof1)) {
      int dofNr = node.GetDofNumber(0);
      node.SetValue(0, vals[dofNr], 0);
      node.SetValue(0, velo[dofNr], 1);
    }
    // **************************
    // Compute
    // **************************
    DofVector<double> loadVector =
        asmbl.BuildVector(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.LoadVector(cipd,
                                [](Eigen::Vector2d coords) { return 0.; });
        });

    Eigen::VectorXd tmp = (-stiffnessMxMod * valsJ);
    tmp += cmat.transpose() *
           (loadVector[dof1] -
            stiffnessMx(dof1, dof1) *
                constraints.GetSparseGlobalRhs(dof1, numDofs, t));

    dwdt.head(w.size() / 2) = veloJ;
    dwdt.tail(w.size() / 2) = (tmp.array() / diagonalMassMod.array()).matrix();
  };

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(domainCellGroup,
                                          NuTo::Visualize::AverageHandler());
    visualize.DofValues(dof1);
    visualize.WriteVtuFile(filename + ".vtu");
  };

  std::cout << "Start time integration" << std::endl;

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    double t = i * stepSize;
    std::cout << i + 1 << std::endl;
    state = ti.DoStep(eq, state, t, stepSize);
    if ((i * 100) % numSteps == 0) {
      visualizeResult("Wave2Dlobatto1smooth_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
