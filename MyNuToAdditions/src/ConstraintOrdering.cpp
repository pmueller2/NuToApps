#include "nuto/mechanics/constraints/Constraints.h"
#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/dofs/DofNumbering.h"

#include "../../NuToHelpers/ConstraintsHelper.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  MeshFem mesh = UnitMeshFem::CreateLines(10);

  DofType dofA("dofA", 1);

  AddDofInterpolation(&mesh, dofA);

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

  Constraint::Constraints constraints;

  Constraint::Equation eq1(ndA_8, 0, [](double) { return 0.; });

  Constraint::Term term1a(ndA_7, 0, -2);
  Constraint::Term term1b(ndA_6, 0, -1);
  eq1.AddIndependentTerm(term1a);
  eq1.AddIndependentTerm(term1b);

  Constraint::Equation eq2(ndA_3, 0, [](double) { return 0.; });

  constraints.Add(dofA, {eq1, eq2});

  int numDofsA = mesh.NodesTotal(dofA).Size();
  DofInfo dofInfo;
  dofInfo.numDependentDofs[dofA] = constraints.GetNumEquations(dofA);
  dofInfo.numIndependentDofs[dofA] = numDofsA - dofInfo.numDependentDofs[dofA];

  // Numbering without paying attention to constraints
  int dofNumber = 0;
  for (auto &node : mesh.NodesTotal(dofA))
    for (int iComponent = 0; iComponent < node.GetNumValues(); ++iComponent)
      node.SetDofNumber(iComponent, dofNumber++);

  Eigen::VectorXi independentGlobalNumbering(dofInfo.numIndependentDofs[dofA]);
  Eigen::VectorXi dependentGlobalNumbering(dofInfo.numDependentDofs[dofA]);

  // ***************************************
  // This is a copy from the constraint matrix stuff
  // ***************************************

  //  // create a vector with all dofs with false:independent true:dependent dof
  std::vector<bool> isDofConstraint(numDofsA, false);
  for (int i = 0; i < constraints.GetNumEquations(dofA); i++) {
    int globalDofNumber =
        constraints.GetEquation(dofA, i).GetDependentDofNumber();
    if (globalDofNumber == -1 /* should be NodeSimple::NOT_SET */)
      throw Exception(__PRETTY_FUNCTION__,
                      "There is no dof numbering for a node in equation" +
                          std::to_string(i) + ".");
    if (globalDofNumber >= numDofsA)
      throw Exception(__PRETTY_FUNCTION__,
                      "The provided dof number of the dependent term exceeds "
                      "the total number of dofs in equation " +
                          std::to_string(i) + ".");
    isDofConstraint[globalDofNumber] = true;
  }

  int independent = 0;
  int dependent = 0;
  for (int i = 0; i < numDofsA; i++) {
    if (isDofConstraint[i]) {
      dependentGlobalNumbering(dependent) = i;
      dependent++;
    } else {
      independentGlobalNumbering(independent) = i;
      independent++;
    }
  }
  assert(independent + dependent == numDofsA);

  std::cout << "independentGlobalNumbering\n"
            << independentGlobalNumbering << std::endl;
  std::cout << "dependentGlobalNumbering\n"
            << dependentGlobalNumbering << std::endl;

  auto numbD =
      Constraint::GetDependentGlobalDofNumbering(constraints, dofA, numDofsA);
  std::cout << "dependentGlobalNumbering\n" << numbD << std::endl;
  auto numbI =
      Constraint::GetIndependentGlobalDofNumbering(constraints, dofA, numDofsA);
  std::cout << "independentGlobalNumbering\n" << numbI << std::endl;
}
