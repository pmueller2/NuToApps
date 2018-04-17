#include "nuto/mechanics/constraints/Constraints.h"
#include "nuto/mechanics/interpolation/InterpolationTriangleLinear.h"
#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/dofs/DofNumbering.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  std::cout << "Introduce optional J,K splitting again" << std::endl;

  DofType dofA("dofA", 1);
  DofType dofB("dofB", 2);

  NuTo::MeshFem mesh = UnitMeshFem::CreateLines(10);

  AddDofInterpolation(&mesh, dofA);
  AddDofInterpolation(&mesh, dofB);

  auto &ndA_0 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.0), dofA);
  auto &ndA_1 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.1), dofA);
  auto &ndA_2 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.2), dofA);
  auto &ndA_3 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.3), dofA);
  auto &ndA_4 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.4), dofA);
  auto &ndA_5 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.5), dofA);
  auto &ndA_6 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.6), dofA);
  auto &ndA_7 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.7), dofA);
  auto &ndA_8 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.8), dofA);
  auto &ndA_9 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.9), dofA);
  auto &ndA_10 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 1.0), dofA);

  auto &ndB_0 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.0), dofB);
  auto &ndB_1 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.1), dofB);
  auto &ndB_2 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.2), dofB);
  auto &ndB_3 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.3), dofB);
  auto &ndB_4 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.4), dofB);
  auto &ndB_5 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.5), dofB);
  auto &ndB_6 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.6), dofB);
  auto &ndB_7 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.7), dofB);
  auto &ndB_8 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.8), dofB);
  auto &ndB_9 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 0.9), dofB);
  auto &ndB_10 = mesh.NodeAtCoordinate(Eigen::VectorXd::Constant(1, 1.0), dofB);

  Constraint::Constraints constraints;

  Constraint::Equation eq1(ndA_8, 0, [](double) { return 0.; });
  Constraint::Term term1a(ndA_7, 0, -2);
  Constraint::Term term1b(ndA_6, 0, -1);
  eq1.AddIndependentTerm(term1a);
  eq1.AddIndependentTerm(term1b);

  constraints.Add(dofA, eq1);

  int numDofsA = mesh.NodesTotal(dofA).Size();
  int numDofsB = mesh.NodesTotal(dofB).Size();

  DofInfo dofInfo;
  dofInfo.numDependentDofs[dofA] = constraints.GetNumEquations(dofA);
  dofInfo.numIndependentDofs[dofA] = numDofsA - dofInfo.numDependentDofs[dofA];

  dofInfo.numDependentDofs[dofB] = constraints.GetNumEquations(dofB);
  dofInfo.numIndependentDofs[dofB] = numDofsB - dofInfo.numDependentDofs[dofB];

  // Numbering without paying attention to constraints
  int dofNumber = 0;
  for (auto &node : mesh.NodesTotal(dofA))
    for (int iComponent = 0; iComponent < node.GetNumValues(); ++iComponent)
      node.SetDofNumber(iComponent, dofNumber++);

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildUnitConstraintMatrix(dofA, numDofsA);
  std::cout << "C-matrix:\n" << cmat << std::endl;

  std::cout << "DofNumbering based on Constraint equations" << std::endl;

  Eigen::VectorXi dependentNumbering(constraints.GetNumEquations(dofA));
  for (int j = 0; j < constraints.GetNumEquations(dofA); j++) {
    dependentNumbering[j] =
        constraints.GetEquation(dofA, j).GetDependentDofNumber();
  }

  Eigen::VectorXi numberingJKsplit(numDofsA);
  int independentCounter = 0;
  for (int i = 0; i < numDofsA; i++) {
    bool isIndependent = true;
    for (int j = 0; j < dependentNumbering.size(); j++) {
      if (i == dependentNumbering[j]) {
        isIndependent = false;
        numberingJKsplit[i] = dofInfo.numIndependentDofs[dofA] + j;
        break;
      }
    }
    if (isIndependent) {
      numberingJKsplit[i] = independentCounter;
      independentCounter++;
    }
  }

  std::cout << numberingJKsplit << std::endl;
}
