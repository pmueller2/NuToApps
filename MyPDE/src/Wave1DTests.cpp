#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/SimpleAssembler.h"

#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"
#include "../../MyTimeIntegration/NY5NoVelocity.h"
#include "../../NuToHelpers/MeshValuesTools.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include "/usr/include/eigen3/unsupported/Eigen/ArpackSupport"

#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include <functional>
#include <iostream>

using namespace NuTo;

void ExtractNodeVals(Eigen::VectorXd &femResult, Group<NodeSimple> nodes) {
  for (NodeSimple &node : nodes) {
    for (int i = 0; i < node.GetNumValues(); i++) {
      int dofNr = node.GetDofNumber(i);
      femResult[dofNr] = node.GetValues()(i);
    }
  }
}

void MergeNodeVals(const Eigen::VectorXd &femResult, Group<NodeSimple> nodes) {
  for (NodeSimple &node : nodes) {
    for (int i = 0; i < node.GetNumValues(); i++) {
      int dofNr = node.GetDofNumber(i);
      node.SetValue(i, femResult[dofNr]);
    }
  };
}

double f1(double x) { return 1. / (5. - 4. * cos(x * 2. * M_PI)); }

double f2(double x, double p = 2.) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  double y = 2. * z - 1.;
  return std::pow((std::abs(1. - std::pow(std::abs(y), p))), (1. / p));
}

double f3(double x) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  return z;
}

double f4(double x, double extent = 0.3) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  z = 2. / extent * z - 1.;
  if ((z < -1.) || (z > 1.))
    return 0.;
  return (0.5 * (1. + cos(M_PI * z)));
}

double f5(double x, double dx) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  z = 2. * z - 1.;
  if (z <= -1. + dx)
    return 0.5 * (1. + sin(0.5 * M_PI * (z + 1.) / dx));
  if (z <= -dx)
    return 1.;
  if (z <= dx)
    return (0.5 * (1. - sin(0.5 * M_PI * z / dx)));
  if (z <= 1. - dx)
    return 0.;
  return 0.5 * (1. + sin(0.5 * M_PI * (z - 1.) / dx));
}

class TestProblem {
public:
  MeshFem mesh;
  const InterpolationSimple &interpolation1D;
  DofType dof1;
  Constraint::Constraints constraints;
  Eigen::SparseMatrix<double> cmat;
  SimpleAssembler asmbl;
  IntegrationTypeTensorProduct<1> integrationType1D;
  Eigen::SparseMatrix<double> stiffnessMxMod;
  Eigen::VectorXd massMxMod;
  Eigen::VectorXd femResult;
  Eigen::VectorXd femVelocities;
  boost::ptr_vector<CellInterface> cells;
  Group<CellInterface> cellGroup;
  int numDofsJ;
  int numDofsK;
  int numDofs;
  int mOrder;
  Group<ElementCollectionFem> elms;
  std::function<double(double)> initialCondition;

  TestProblem(int numElms, int order)
      : mesh(std::move(UnitMeshFem::CreateLines(numElms))),
        interpolation1D(
            mesh.CreateInterpolation(InterpolationTrussLobatto(order))),
        dof1("Scalar", 1),
        integrationType1D(order + 1, eIntegrationMethod::LOBATTO),
        mOrder(order), elms(mesh.ElementsTotal()) {

    AddDofInterpolation(&mesh, dof1, interpolation1D);
    NodeSimple &nd0 =
        mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.), dof1);
    NodeSimple &nd1 =
        mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 1.), dof1);

    Constraint::Term term2(nd1, 0, -1);
    Constraint::Equation constraintEq(nd0, 0, [](double) { return 0.; });
    constraintEq.AddTerm(term2);
    constraints.Add(dof1, constraintEq);

    DofInfo dofInfo =
        DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

    numDofsJ = dofInfo.numIndependentDofs[dof1];
    numDofsK = dofInfo.numDependentDofs[dof1];
    numDofs = numDofsJ + numDofsK;

    cmat = constraints.BuildConstraintMatrix(dof1, numDofsJ);
    asmbl = SimpleAssembler(dofInfo);

    int cellId = 0;
    for (ElementCollection &element : mesh.Elements) {
      cells.push_back(new Cell(element, integrationType1D, cellId++));
    }
    for (CellInterface &c : cells) {
      cellGroup.Add(c);
    }

    // Matrix assembly
    Integrands::PoissonTypeProblem<1> equations(dof1);

    auto stiffnessMx = asmbl.BuildMatrix(
        cellGroup, {dof1},
        [&](const CellIpData cipd) { return equations.StiffnessMatrix(cipd); });
    auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
        cellGroup, {dof1},
        [&](const CellIpData cipd) { return equations.MassMatrix(cipd); });

    // Modify because of constraints
    auto kJJ = stiffnessMx.JJ(dof1, dof1);
    auto kJK = stiffnessMx.JK(dof1, dof1);
    auto kKJ = stiffnessMx.KJ(dof1, dof1);
    auto kKK = stiffnessMx.KK(dof1, dof1);

    stiffnessMxMod = kJJ - cmat.transpose() * kKJ - kJK * cmat +
                     cmat.transpose() * kKK * cmat;

    auto mJ = lumpedMassMx.J[dof1];
    auto mK = lumpedMassMx.K[dof1];

    Eigen::SparseMatrix<double> massModifier =
        cmat.transpose() * mK.asDiagonal() * cmat;

    massMxMod = mJ + massModifier.diagonal();

    // Setup a solution vector
    femResult.resize(numDofs);
    femResult.setZero();
    femVelocities.resize(numDofs);
    femVelocities.setZero();
  }

  void Solve(int numSteps, double stepSize) {

    auto problemToSolve = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2,
                              double t) {
      Eigen::VectorXd tmp = -stiffnessMxMod * w;
      d2wdt2 = (tmp.array() / massMxMod.array()).matrix();
    };

    ExtractNodeVals(femResult, mesh.NodesTotal(dof1));

    auto state =
        std::make_pair(femResult.head(numDofsJ), femVelocities.head(numDofsJ));

    NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;

    for (int i = 0; i < numSteps; i++) {
      double t = i * stepSize;
      state = ti.DoStep(problemToSolve, state.first, state.second, t, stepSize);
      femResult.head(numDofsJ) = state.first;
      femResult.tail(numDofsK) =
          -cmat * state.first + constraints.GetRhs(dof1, (i + 1) * stepSize);
      femVelocities.head(numDofsJ) = state.second;
      femVelocities.tail(numDofsK) = -cmat * state.second;
    }
    MergeNodeVals(femResult, mesh.NodesTotal(dof1));
  }

  double LargestEigenvalue() {
    std::string typeOfVal = "LM"; // Largest magnitude

    Eigen::SparseMatrix<double> M =
        Eigen::MatrixXd(massMxMod.asDiagonal()).sparseView();

    Eigen::ArpackGeneralizedSelfAdjointEigenSolver<Eigen::SparseMatrix<double>>
        eigSolver(stiffnessMxMod, M, 1, typeOfVal);

    if (eigSolver.info() != Eigen::Success) {
      throw Exception("ArpackEigensolver did not succeed!");
    }

    return eigSolver.eigenvalues()[0];
  }

  void SetInitialCondition(std::function<double(double)> f) {
    initialCondition = f;
    Tools::SetValues(elms, dof1, f);
  }

  void Plot(std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        cellGroup, NuTo::Visualize::VoronoiHandler(
                       NuTo::Visualize::VoronoiGeometryLine(5 * mOrder)));
    visualize.PointData(
        [&](Eigen::VectorXd x) {
          return Eigen::VectorXd::Constant(1, initialCondition(x[0]));
        },
        "InitialCondition");

    visualize.DofValues(dof1);
    visualize.WriteVtuFile(filename + ".vtu");
  }

  double L2Error() {
    auto ErrorL2Func = [&](const CellIpData &cipd) {

      Eigen::VectorXd localVal = cipd.Value(dof1);
      double error =
          localVal[0] - initialCondition(cipd.GlobalCoordinates()[0]);
      return (error * error);
    };

    // Create cells for more accurate integration
    int cellId = 0;
    IntegrationTypeTensorProduct<1> igType(10 * mOrder,
                                           eIntegrationMethod::GAUSS);
    boost::ptr_vector<CellInterface> errorcells;
    Group<CellInterface> errorcellGroup;
    for (ElementCollection &element : mesh.Elements) {
      errorcells.push_back(new Cell(element, igType, cellId++));
    }
    for (CellInterface &c : cells) {
      errorcellGroup.Add(c);
    }

    double L2Error = 0.;
    for (CellInterface &cell : errorcellGroup) {
      L2Error += sqrt(cell.Integrate(ErrorL2Func));
    }
    L2Error /= errorcellGroup.Size();
    return L2Error;
  }

  double MaxError() {
    double maxError = 0.;

    auto ErrorMaxFunc = [&](const CellIpData &cipd) {

      Eigen::VectorXd localVal = cipd.Value(dof1);
      double error =
          localVal[0] - initialCondition(cipd.GlobalCoordinates()[0]);
      return (std::max(maxError, std::abs(error)));
    };

    // Create cells for more accurate integration
    int cellId = 0;
    IntegrationTypeTensorProduct<1> igType(10 * mOrder,
                                           eIntegrationMethod::GAUSS);
    boost::ptr_vector<CellInterface> errorcells;
    Group<CellInterface> errorcellGroup;
    for (ElementCollection &element : mesh.Elements) {
      errorcells.push_back(new Cell(element, igType, cellId++));
    }
    for (CellInterface &c : errorcells) {
      errorcellGroup.Add(c);
    }

    for (CellInterface &cell : errorcellGroup) {
      double errorResult = cell.Integrate(ErrorMaxFunc);
      maxError = errorResult;
    }
    return maxError;
  }
};

int main(int argc, char *argv[]) {
  int numCycles = 10;
  double stepSize = 0.0001;
  int numSteps = 10000 * numCycles;

  for (int i = 20; i < 201; i++) {
    int numElms = i;
    int order = 1;
    int numDofs = numElms * order;

    TestProblem pr(numElms, order);
    pr.SetInitialCondition([&](double x) { return f5(x, 0.1); });
    pr.Solve(numSteps, stepSize);
    pr.Plot("Wave1d");
    std::cout << numDofs;
    std::cout << "    " << pr.L2Error();
    std::cout << "   " << pr.MaxError();
    std::cout << "   " << sqrt(pr.LargestEigenvalue());
    std::cout << std::endl;
  }
}
