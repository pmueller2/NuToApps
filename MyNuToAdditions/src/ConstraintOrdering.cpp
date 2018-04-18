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

  ndA_0.SetValue(0, 0.0);
  ndA_1.SetValue(0, 0.1);
  ndA_2.SetValue(0, 0.2);
  ndA_3.SetValue(0, 0.3);
  ndA_4.SetValue(0, 0.4);
  ndA_5.SetValue(0, 0.5);
  ndA_6.SetValue(0, 0.6);
  ndA_7.SetValue(0, 0.7);
  ndA_8.SetValue(0, 0.8);
  ndA_9.SetValue(0, 0.9);
  ndA_10.SetValue(0, 1.0);

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
  ndA_0.SetDofNumber(0, 0);
  ndA_1.SetDofNumber(0, 1);
  ndA_2.SetDofNumber(0, 2);
  ndA_3.SetDofNumber(0, 3);
  ndA_4.SetDofNumber(0, 4);
  ndA_5.SetDofNumber(0, 5);
  ndA_6.SetDofNumber(0, 6);
  ndA_7.SetDofNumber(0, 7);
  ndA_8.SetDofNumber(0, 8);
  ndA_9.SetDofNumber(0, 9);
  ndA_10.SetDofNumber(0, 10);

  // Extract values
  Eigen::VectorXd values(11);
  values.setOnes();
  values *= 42;
  for (NodeSimple &nd : mesh.NodesTotal(dofA)) {
    values(nd.GetDofNumber(0)) = nd.GetValues()(0);
  }
  std::cout << "Values" << values << std::endl;

  auto numbering = Constraint::GetJKNumbering(constraints, dofA, numDofsA);
  std::cout << "independentGlobalNumbering\n"
            << numbering.head(numDofsA - constraints.GetNumEquations(dofA))
            << std::endl;
  std::cout << "dependentGlobalNumbering\n"
            << numbering.tail(constraints.GetNumEquations(dofA)) << std::endl;

  Eigen::PermutationMatrix<Eigen::Dynamic> P(numbering);
  std::cout << "Global to JK ordering\n" << P.transpose() * values << std::endl;

  Eigen::MatrixXd M1(11, 11);
  Eigen::MatrixXd M2(11, 11);
  for (int i = 0; i < 11; i++) {
    for (int j = 0; j < 11; j++) {
      M1(i, j) = i;
      M2(i, j) = j;
    }
  }

  std::cout << "M1 \n" << M1 << std::endl;
  std::cout << "M1 Global to JK ordering\n"
            << P.transpose() * M1 * P << std::endl;

  std::cout << "M2 \n" << M1 << std::endl;
  std::cout << "M2 Global to JK ordering\n"
            << P.transpose() * M2 * P << std::endl;
}
