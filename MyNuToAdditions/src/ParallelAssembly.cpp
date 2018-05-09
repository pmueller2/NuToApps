#include "nuto/base/Timer.h"
#include "nuto/mechanics/cell/CellIpData.h"
#include "nuto/mechanics/constraints/Constraints.h"
#include "nuto/mechanics/dofs/DofNumbering.h"
#include "nuto/mechanics/dofs/DofVector.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"
#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"
#include "nuto/mechanics/tools/CellStorage.h"
#include "test/tools/BoostUnitTest.h"

#include <iostream>

using namespace NuTo;

BOOST_AUTO_TEST_CASE(OmpParallelVectorAssembly) {
  int n = 1e4;
  int order = 4;
  MeshFem mesh;
  {
    Timer timer("UnitMeshFem::CreateLines");
    mesh = UnitMeshFem::CreateLines(n);
  }
  DofType dof("Displacement", 1);
  {
    Timer timer("AddDofInterpolation");
    AddDofInterpolation(
        &mesh, dof, mesh.CreateInterpolation(InterpolationTrussLobatto(order)));
  }

  IntegrationTypeTensorProduct<1> integrationType(order + 1,
                                                  eIntegrationMethod::LOBATTO);
  Group<ElementCollectionFem> domain;
  {
    Timer timer("ElementsTotal");
    domain = mesh.ElementsTotal();
  }
  Group<NodeSimple> dofNodes;
  {
    Timer timer("NodesTotal");
    dofNodes = mesh.NodesTotal(dof);
  }
  CellStorage cells;
  Group<CellInterface> cellGroup = cells.AddCells(domain, integrationType);

  auto rightHandSide = [&](const CellIpData &cipd) {
    Eigen::MatrixXd B = cipd.B(dof, Nabla::Gradient());
    DofVector<double> result;
    result[dof] = -B.transpose() * B * cipd.NodeValueVector(dof);
    return result;
  };

  // ******************************************
  // Assembly
  // ******************************************

  Constraint::Constraints constraints;
  DofInfo dofInfo = DofNumbering::Build(dofNodes, dof, constraints);
  int numDofs = dofInfo.numDependentDofs[dof] + dofInfo.numIndependentDofs[dof];

  Timer timer("Assembly parallel");

  int numThreads = 4;

  Eigen::VectorXd result(numDofs);

#pragma omp parallel num_threads(numThreads) default(shared) firstprivate(dof)
  {
    Eigen::VectorXd gradient(numDofs);
    gradient.setZero();
#pragma omp for nowait
    for (auto it = cellGroup.begin(); it < cellGroup.end(); it++) {
      const Eigen::VectorXd cellGradient = it->Integrate(rightHandSide)[dof];

      Eigen::VectorXi numberingDof = it->DofNumbering(dof);
      for (int i = 0; i < numberingDof.rows(); ++i)
        gradient(numberingDof[i]) += cellGradient[i];
    }
#pragma omp critical
    result += gradient;
  }

  timer.Reset("Assembly single core");

  // ******************************************
  // Check Assembly by not using omp
  // ******************************************

  Eigen::VectorXd singleCoreresult(numDofs);
  singleCoreresult.setZero();
  for (auto it = cellGroup.begin(); it < cellGroup.end(); it++) {
    const Eigen::VectorXd cellGradient = it->Integrate(rightHandSide)[dof];

    Eigen::VectorXi numberingDof = it->DofNumbering(dof);
    for (int i = 0; i < numberingDof.rows(); ++i)
      singleCoreresult(numberingDof[i]) += cellGradient[i];
  }

  timer.Reset("Matrix comparison");
  BoostUnitTest::CheckEigenMatrix(result, singleCoreresult);
}

BOOST_AUTO_TEST_CASE(OmpParallelMatrixAssembly) {
  int n = 1e4;
  int order = 4;
  MeshFem mesh = UnitMeshFem::CreateLines(n);
  DofType dof("Displacement", 1);
  AddDofInterpolation(
      &mesh, dof, mesh.CreateInterpolation(InterpolationTrussLobatto(order)));
  IntegrationTypeTensorProduct<1> integrationType(order + 1,
                                                  eIntegrationMethod::LOBATTO);
  Group<ElementCollectionFem> domain = mesh.ElementsTotal();
  Group<NodeSimple> dofNodes = mesh.NodesTotal(dof);
  CellStorage cells;
  Group<CellInterface> cellGroup = cells.AddCells(domain, integrationType);

  auto stiffness = [dof](const CellIpData &cipd) {
    Eigen::MatrixXd B = cipd.B(dof, Nabla::Gradient());
    DofMatrix<double> stiffnessLocal;
    stiffnessLocal(dof, dof) = B.transpose() * 1 * B;
    return stiffnessLocal;
  };

  // ******************************************
  // Assembly
  // ******************************************

  Constraint::Constraints constraints;
  DofInfo dofInfo = DofNumbering::Build(dofNodes, dof, constraints);
  int numDofs = dofInfo.numDependentDofs[dof] + dofInfo.numIndependentDofs[dof];

  Timer timer("Matrix Assembly parallel");

  int numThreads = 4;

  Eigen::SparseMatrix<double> hessian(numDofs, numDofs);
  std::list<Eigen::Triplet<double>> triplets;

#pragma omp parallel num_threads(numThreads) default(shared) firstprivate(dof)
  {
    std::list<Eigen::Triplet<double>> localtriplets;
#pragma omp for nowait
    for (auto it = cellGroup.begin(); it < cellGroup.end(); it++) {
      const Eigen::MatrixXd cellHessian = it->Integrate(stiffness)(dof, dof);
      Eigen::VectorXi numberingDof = it->DofNumbering(dof);
      for (int i = 0; i < numberingDof.rows(); ++i) {
        for (int j = 0; j < numberingDof.rows(); ++j) {
          const int globalDofNumberI = numberingDof[i];
          const int globalDofNumberJ = numberingDof[j];
          const double globalDofValue = cellHessian(i, j);
          localtriplets.push_back(
              {globalDofNumberI, globalDofNumberJ, globalDofValue});
        }
      }
    }
#pragma omp critical
    triplets.splice(triplets.end(), localtriplets);
  }

  {
    timer.Reset("    SetFromTriplets");

    hessian.setFromTriplets(triplets.begin(), triplets.end());
  }
  // ******************************************
  // Check Assembly by not using omp
  // ******************************************

  timer.Reset("Matrix Assembly single core");

  Eigen::SparseMatrix<double> hessianSingleCore(numDofs, numDofs);
  std::list<Eigen::Triplet<double>> tripletsSingleCore;

  for (auto it = cellGroup.begin(); it < cellGroup.end(); it++) {
    const Eigen::MatrixXd cellHessian = it->Integrate(stiffness)(dof, dof);
    Eigen::VectorXi numberingDof = it->DofNumbering(dof);
    for (int i = 0; i < numberingDof.rows(); ++i) {
      for (int j = 0; j < numberingDof.rows(); ++j) {
        const int globalDofNumberI = numberingDof[i];
        const int globalDofNumberJ = numberingDof[j];
        const double globalDofValue = cellHessian(i, j);
        tripletsSingleCore.push_back(
            {globalDofNumberI, globalDofNumberJ, globalDofValue});
      }
    }
  }
  hessianSingleCore.setFromTriplets(tripletsSingleCore.begin(),
                                    tripletsSingleCore.end());

  timer.Reset("Matrix comparison");
  double diffNorm = (hessian - hessianSingleCore).norm();

  BOOST_CHECK_SMALL(diffNorm, 1e-6);
}
