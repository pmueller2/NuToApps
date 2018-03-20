#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include <functional>
#include <iostream>

using namespace NuTo;

/*  Solves Wave equation on a Line:
 *  d2u/dt2 - d2u/dx2 = f
 *
 * The corresponding FEM equation is then
 *
 * Mu + Ku = f + bndry
 *
 * Here K is the stiffness mx:    int (grad v)(grad u) dV
 *      M is the mass mx     :    int vu dV
 *      f load vector        :    int vf dV
 *      bndry boundary term  :    int vg dA
 *
 *
 * */
int main(int argc, char *argv[]) {

  // **************************************
  //      Set some problem parameters
  // **************************************

  double stepSize = 0.002;
  int numSteps = 500;
  // ***************************
  //      Generate a mesh
  // ***************************

  int numElms = 2;
  int order = 12;
  MeshFem mesh = UnitMeshFem::CreateLines(numElms);

  // ***************************
  //      Add DoFs
  // ***************************

  const InterpolationSimple &interpolation1D =
      mesh.CreateInterpolation(InterpolationTrussLobatto(order));

  DofType dof1("Scalar", 1);
  AddDofInterpolation(&mesh, dof1, interpolation1D);

  DofType dof2("Exact", 1);
  AddDofInterpolation(&mesh, dof2, interpolation1D);

  // ******************************
  //      Add boundary information
  // ******************************

  NodeSimple &nd0 =
      mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.), dof1);
  NodeSimple &nd1 =
      mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 1.), dof1);

  std::map<NodeSimple *, double> nodeCoordinateMap;
  nodeCoordinateMap[&nd0] = 0.;
  nodeCoordinateMap[&nd1] = 1.;
  std::map<NodeSimple *, double> nodeSurfaceNormalMap;
  nodeSurfaceNormalMap[&nd0] = -1;
  nodeSurfaceNormalMap[&nd1] = 1.;

  Group<NodeSimple> dirichletBoundary = {};
  Group<NodeSimple> neumannBoundary = {nd0, nd1};
  bool periodic = true;

  if (periodic) {
    neumannBoundary = Group<NodeSimple>({});
  }

  // ***********************************
  //   List of some possible solutions
  // ***********************************

  auto solution = [](double r, double t) {
    if (((r - t) > 0) && ((r - t) < 1))
      return 0.5 * (1 - cos(2 * M_PI * (r - t)));
    return 0.;
  };
  auto solutionSpaceDerivative = [](double r, double t) {
    //    if (((r - t) > 0) && ((r - t) < 1))
    //      return M_PI * sin(2 * M_PI * (r - t));
    return 0.;
  };
  auto solutionTimeDerivative = [](double r, double t) {
    if (((r - t) > 0) && ((r - t) < 1))
      return -M_PI * sin(2 * M_PI * (r - t));
    return 0.;
  };

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  Constraint::Constraints constraints;

  if (periodic) {
    Constraint::Term term2(nd1, 0, -1);
    Constraint::Equation constraintEq(nd0, 0, [](double) { return 0.; });
    constraintEq.AddTerm(term2);
    constraints.Add(dof1, constraintEq);
  } else {

    for (auto &nd : dirichletBoundary) {
      constraints.Add(dof1, Constraint::Value(nd, [&](double t) {
                        return solution(nodeCoordinateMap.at(&nd), t);
                      }));
    }
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

  IntegrationTypeTensorProduct<1> integrationType1D(
      order + 1, eIntegrationMethod::LOBATTO);

  // volume cells
  boost::ptr_vector<CellInterface> cells;
  int cellId = 0;
  for (ElementCollection &element : mesh.Elements) {
    cells.push_back(new Cell(element, integrationType1D, cellId++));
  }
  Group<CellInterface> cellGroup;
  for (CellInterface &c : cells) {
    cellGroup.Add(c);
  }

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  using namespace std::placeholders;

  Integrands::PoissonTypeProblem<1> pde(dof1);

  auto stiffnessMx = asmbl.BuildMatrix(
      cellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.StiffnessMatrix(cipd); });

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      cellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.MassMatrix(cipd); });

  auto loadVector = asmbl.BuildVector(
      cellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.LoadVector(cipd); });

  // Compute modified stiffness matrix
  auto kJJ = stiffnessMx.JJ(dof1, dof1);
  auto kJK = stiffnessMx.JK(dof1, dof1);
  auto kKJ = stiffnessMx.KJ(dof1, dof1);
  auto kKK = stiffnessMx.KK(dof1, dof1);

  Eigen::SparseMatrix<double> stiffnessMxMod =
      kJJ - cmat.transpose() * kKJ - kJK * cmat + cmat.transpose() * kKK * cmat;

  // ***********************************
  //    Add boundary loads
  // ***********************************

  auto addBoundaryLoads = [&](double t) {
    GlobalDofVector boundaryLoadVector = loadVector;
    for (auto &nd : neumannBoundary) {
      double normal = nodeSurfaceNormalMap.at(&nd);
      double coord = nodeCoordinateMap.at(&nd);
      boundaryLoadVector.J[dof1](nd.GetDofNumber(0)) +=
          normal * solutionSpaceDerivative(coord, t);
    };
    return boundaryLoadVector;
  };

  auto computeModifiedLoadVector = [&](double t) {
    auto tmpLoad = addBoundaryLoads(t);
    auto fJ = tmpLoad.J[dof1];
    auto fK = tmpLoad.K[dof1];
    auto b = constraints.GetRhs(dof1, t);
    Eigen::VectorXd loadVectorMod = fJ - cmat.transpose() * fK;
    loadVectorMod -= (kJK * b - cmat.transpose() * kKK * b);
    return loadVectorMod;
  };

  // ***********************************
  //    Set up initial condition
  // ***********************************

  Eigen::VectorXd femVelocities(dofInfo.numIndependentDofs[dof1] +
                                dofInfo.numDependentDofs[dof1]);

  // DOF 1 (initial condition)
  for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
    NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
    NuTo::ElementFem &elmDof = elmColl.DofElement(dof1);
    for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
      Eigen::VectorXd coord =
          Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      elmDof.GetNode(i).SetValue(0, solution(coord(0), 0.));
      femVelocities[elmDof.GetNode(i).GetDofNumber(0)] =
          solutionTimeDerivative(coord(0), 0);
    }
  }

  auto mergeExactSolution = [&](double t) {
    for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof2);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        Eigen::VectorXd coord =
            Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        elmDof.GetNode(i).SetValue(0, solution(coord(0), t));
      }
    };
  };

  mergeExactSolution(0);

  // ***********************************
  //    Visualize
  // ***********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        cellGroup, NuTo::Visualize::VoronoiHandler(
                       NuTo::Visualize::VoronoiGeometryLine(5 * order)));
    visualize.DofValues(dof1);
    visualize.DofValues(dof2);
    visualize.WriteVtuFile(filename + ".vtu");
  };

  visualizeResult("WaveLine_0");

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
  Eigen::VectorXd w0 = femResult.head(dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd v0 = femVelocities.head(dofInfo.numIndependentDofs[dof1]);

  auto MergeResult = [&]() {
    for (auto &node : mesh.NodesTotal(dof1)) {
      int dofNr = node.GetDofNumber(0);
      node.SetValue(0, femResult[dofNr]);
    };
  };

  auto state = std::make_pair(w0, v0);

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
    Eigen::VectorXd tmp = (-stiffnessMxMod * w + computeModifiedLoadVector(t));
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
      MergeResult();
      mergeExactSolution((i + 1) * stepSize);
      visualizeResult("WaveLine_" + std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
