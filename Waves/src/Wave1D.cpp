#include "../../MyTimeIntegration/RK4.h"
#include "../../NuToHelpers/BoostOdeintEigenSupport.h"
#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/MeshValuesTools.h"
#include "nuto/mechanics/cell/CellIpData.h"
#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/constraints/Constraints.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"
#include "nuto/mechanics/tools/CellStorage.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"

#include "boost/filesystem.hpp"
#include "boost/numeric/odeint/stepper/runge_kutta4.hpp"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  int numElms = 100;
  int interpolationOrder = 3;
  int numIPs = interpolationOrder + 1;

  int numSteps = 1000;
  double stepSize = 0.001;
  double tau = 0.500;

  // **************************************
  // Result directory, filesystem
  // **************************************

  std::string resultDirectory = "/Wave1D/";
  bool overwriteResultDirectory = true;

  // delete result directory if it exists and create it new
  boost::filesystem::path rootPath = boost::filesystem::initial_path();
  boost::filesystem::path resultDirectoryFull = rootPath.parent_path()
                                                    .parent_path()
                                                    .append("/results")
                                                    .append(resultDirectory);

  if (boost::filesystem::exists(resultDirectoryFull)) // does p actually exist?
  {
    if (boost::filesystem::is_directory(resultDirectoryFull)) {
      if (overwriteResultDirectory) {
        boost::filesystem::remove_all(resultDirectoryFull);
        boost::filesystem::create_directory(resultDirectoryFull);
      }
    }
  } else {
    boost::filesystem::create_directory(resultDirectoryFull);
  }

  // **************************************
  // Mesh, Dofs, Interpolation, Numbering
  // **************************************

  MeshFem mesh = UnitMeshFem::CreateLines(numElms);
  DofType dof("scalar", 1);
  InterpolationTrussLobatto ipol(interpolationOrder);
  AddDofInterpolation(&mesh, dof, mesh.CreateInterpolation(ipol));
  mesh.AllocateDofInstances(dof, 2);
  auto domain = mesh.ElementsTotal();
  auto dofNodes = mesh.NodesTotal(dof);

  NodeSimple &leftDofNode =
      mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.), dof);
  NodeSimple &rightDofNode =
      mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 1.), dof);

  int numDofs = dofNodes.Size();

  int counter = 0;
  for (auto &nd : dofNodes) {
    nd.SetDofNumber(0, counter);
    counter++;
  }

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  IntegrationTypeTensorProduct<1> integrationType(numIPs,
                                                  eIntegrationMethod::LOBATTO);
  CellStorage domainCells;
  auto domainCellGroup = domainCells.AddCells(domain, integrationType);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  Constraint::Constraints constraints;

  constraints.Add(dof, Constraint::Equation(leftDofNode, 0, [tau](double t) {
                    if ((0 < t) && (t < tau)) {
                      return 0.5 * (1. - cos(2 * M_PI * t / tau));
                    }
                    return 0.;
                  }));
  constraints.Add(
      dof, Constraint::Equation(rightDofNode, 0, [](double t) { return 0.; }));

  //  Constraint::Equation periodic(leftDofNode, 0, [](double t) { return 0.;
  //  });
  //  periodic.AddIndependentTerm(Constraint::Term(rightDofNode, 0, -1));
  //  constraints.Add(dof, periodic);

  int numDofsK = constraints.GetNumEquations(dof);
  int numDofsJ = numDofs - numDofsK;
  DofInfo dofInfo;
  dofInfo.numIndependentDofs[dof] = numDofsJ;
  dofInfo.numDependentDofs[dof] = numDofsK;

  auto cmat = constraints.BuildUnitConstraintMatrix(dof, numDofs);
  Eigen::PermutationMatrix<Eigen::Dynamic> P(
      Constraint::GetJKNumbering(constraints, dof, numDofs));

  // ******************************
  //    Compute Mass Matrix
  // ******************************

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  Eigen::MatrixXd lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof}, [dof](const CellIpData &cipd) {
        Eigen::MatrixXd N = cipd.N(dof);
        DofMatrix<double> massLocal;
        massLocal(dof, dof) = N.transpose() * N;
        return massLocal;
      })[dof];

  Eigen::VectorXd massMxMod =
      (cmat.transpose() * lumpedMassMx.asDiagonal() * cmat).eval().diagonal();
  Eigen::VectorXd massMxModInv2 =
      ((cmat * massMxMod.asDiagonal().inverse() * cmat.transpose()).eval())
          .diagonal();

  Eigen::SparseMatrix<double> massMxModInv2Full =
      cmat * massMxMod.asDiagonal().inverse() * cmat.transpose();

  Eigen::VectorXd lumpedMassMxInv =
      lumpedMassMx.asDiagonal().inverse().diagonal();

  // ******************************
  //    Define right hand side
  // ******************************

  auto rightHandSide = [dof](const CellIpData &cipd) {
    Eigen::MatrixXd B = cipd.B(dof, Nabla::Gradient());
    DofVector<double> result;
    result[dof] = -B.transpose() * B * cipd.NodeValueVector(dof);
    return result;
  };

  // ******************************
  //    Set up equation system
  // ******************************

  auto ODESystem1stOrder = [&](const Eigen::VectorXd &w, Eigen::VectorXd &dwdt,
                               double t) {
    // JK ordering
    Eigen::VectorXd valsJ = (P.transpose() * w.head(numDofs)).head(numDofsJ);
    Eigen::VectorXd veloJ = (P.transpose() * w.tail(numDofs)).head(numDofsJ);
    // Update constrained dofs
    auto B = constraints.GetSparseGlobalRhs(dof, numDofs, t);
    Eigen::VectorXd vals = cmat * valsJ + B;
    Eigen::VectorXd velo = cmat * veloJ; // Here is actually a dot(B) missing
    // Node Merge
    for (auto &nd : dofNodes) {
      int dofNr = nd.GetDofNumber(0);
      nd.SetValue(0, vals[dofNr], 0);
      nd.SetValue(0, velo[dofNr], 1);
    }
    Eigen::VectorXd rhsFull =
        asmbl.BuildVector(domainCellGroup, {dof}, rightHandSide)[dof];
    dwdt.head(numDofs) = velo;
    Eigen::VectorXd dwdtJK = massMxModInv2Full * rhsFull;
    dwdt.tail(numDofs) = dwdtJK;
  };

  // ******************************
  //      Set Initial Condition
  // ******************************

  // Dof Values
  auto cosineBump = [](double x) {
    double a = 0.3;
    double b = 0.7;
    if ((a < x) && (x < b)) {
      return 0.5 * (1. - cos(2 * M_PI * (x - a) / (b - a)));
    }
    return 0.;
  };

  double speed = -1.;

  auto cosineBumpDerivative = [speed](double x) {
    double a = 0.3;
    double b = 0.7;
    if ((a < x) && (x < b)) {
      return speed * M_PI / (b - a) * sin(2 * M_PI * (x - a) / (b - a));
    }
    return 0.;
  };

  // Velocities
  auto zeroFunc = [](double x) { return 0.; };

  Tools::SetValues(domain, dof, zeroFunc, 0);
  Tools::SetValues(domain, dof, zeroFunc, 1);

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        domainCellGroup,
        NuTo::Visualize::VoronoiHandler(
            Visualize::VoronoiGeometryLine(numIPs, Visualize::LOBATTO)));
    visualize.DofValues(dof);
    visualize.WriteVtuFile(filename + ".vtu");
  };

  // ******************************
  //   Time integration
  // ******************************

  Eigen::VectorXd state(2 * numDofs);
  // Extract values
  for (NodeSimple &nd : dofNodes) {
    int dofNr = nd.GetDofNumber(0);
    state[dofNr] = nd.GetValues(0)[0];
    state[dofNr + numDofs] = nd.GetValues(1)[0];
  }

  TimeIntegration::RK4<Eigen::VectorXd> ti;

  //  boost::numeric::odeint::runge_kutta4<
  //      Eigen::VectorXd, double, Eigen::VectorXd, double,
  //      boost::numeric::odeint::vector_space_algebra>
  //      stepper;

  double t = 0.;
  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * stepSize;
    state = ti.DoStep(ODESystem1stOrder, state, t, stepSize);

    // For odeint constant steppers:
    // stepper.do_step(ODESystem1stOrder, state, t, stepSize);

    std::cout << i + 1 << std::endl;
    if ((i * 100) % numSteps == 0) {
      visualizeResult(resultDirectoryFull.string() + std::string("Wave1D_") +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
