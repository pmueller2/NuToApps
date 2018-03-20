#include "boost/ptr_container/ptr_vector.hpp"
#include <cmath>
#include <iostream>

#include "mechanics/mesh/MeshFem.h"
#include "mechanics/mesh/MeshFemDofConvert.h"
#include "mechanics/mesh/UnitMeshFem.h"

#include "mechanics/cell/SimpleAssember.h"
#include "mechanics/dofs/DofNumbering.h"

#include "mechanics/cell/Cell.h"
#include "mechanics/cell/CellInterface.h"

#include "mechanics/constraints/ConstraintCompanion.h"
#include "mechanics/constraints/Constraints.h"

#include "mechanics/interpolation/InterpolationBrickLinear.h"
#include "mechanics/interpolation/InterpolationBrickLobatto.h"
#include "mechanics/interpolation/InterpolationQuadLinear.h"
#include "mechanics/interpolation/InterpolationTrussLinear.h"
#include "mechanics/interpolation/InterpolationTrussLobatto.h"

#include "mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "visualize/AverageGeometries.h"
#include "visualize/AverageHandler.h"
#include "visualize/UnstructuredGrid.h"
#include "visualize/Visualizer.h"
#include "visualize/VoronoiGeometries.h"
#include "visualize/VoronoiHandler.h"
#include "visualize/XMLWriter.h"

#include "../MyTimeIntegration/ImplicitEuler.hpp"
#include "../MyTimeIntegration/NY4NoVelocity.h"

double smearedStepFunction(double t, double tau) {
  double omega = M_PI / tau;
  double ot = omega * t;
  double result = 0.5 * (1. - cos(ot));
  if (ot > M_PI) {
    result = 1.;
  } else if (ot < 0.) {
    result = 0.;
  }
  return result;
}

using namespace NuTo;

/* Represents a Poisson type PDE problem in 1D/2D/3D
 * i.e. an equation in the form
 *
 * (Dt)u = Lu + f
 *
 * with Lu = div(K * grad u), in general a matrix K(dim,dim)
 * given source function f
 * and time derivative operator Dt as follows:
 *
 * Laplace/Poisson equation Dt = 0
 * Diffusion (Dt)u = d/dt u
 * Wave eq.  (Dt)u = d2/dt2 u
 * Helmholtz (Dt)u = u
 *
 * Combinations include the telegraph equation (linear)
 * or nonlinear variants like fishers equation (diffusion with nonlinear source)
 * or burgers equation (nonlinear advection)
 *
 * */
class PoissonTypeProblem {
public:
  int order;
  double rho;
  int numElm;
  MeshFem mesh;
  DofType dof;
  double riseTime;
  Constraint::Constraints constraints;
  Eigen::SparseMatrix<double> cmat;
  DofNumbering::DofInfo dofInfo;
  boost::ptr_vector<CellInterface> integrationCells;
  Group<CellInterface> allCells;
  SimpleAssembler asmbl;
  GlobalDofMatrixSparse massMx;
  GlobalDofMatrixSparse stiffnessMx;
  GlobalDofVector lumpedMassMx;
  GlobalDofVector loadVector;
  DofMatrixSparse<double> massMxMod;
  DofVector<double> lumpedMassMxMod;
  DofMatrixSparse<double> stiffnessMxMod;
  NuTo::DofVector<double> loadVectorMod;
  IntegrationTypeTensorProduct<1> integrationType1D;
  IntegrationTypeTensorProduct<2> integrationType2D;
  IntegrationTypeTensorProduct<3> integrationType3D;
  Eigen::VectorXd femResult;
  Eigen::MatrixXd systemMx;
  std::map<NodeSimple *, Eigen::VectorXd> nodesAndCoordinates;

  PoissonTypeProblem(int ord, int nElm)
      : order(ord), rho(1.), numElm(nElm), dof("Scalar", 1), riseTime(0.1),
        asmbl(dofInfo.numIndependentDofs, dofInfo.numDependentDofs),
        integrationType1D(order + 1, eIntegrationMethod::LOBATTO),
        integrationType2D(2, eIntegrationMethod::LOBATTO),
        integrationType3D(2, eIntegrationMethod::LOBATTO) {}

  //! Adds a 1D mesh with linear elements in range [0,1]
  void AddMesh1D(int numElm) { mesh = NuTo::UnitMeshFem::CreateLines(numElm); }

  //! Adds a 2D mesh with quad elements in range [0,1] * [0,1]
  void AddMesh2D(int numElm) {
    mesh = NuTo::UnitMeshFem::CreateQuads(numElm, numElm);
  }

  //! Adds a 1D unit mesh boundary
  void AddMesh1DBoundary() {}

  void FillNodesAndCoordinatesMap() {
    // Get Coordinates of nodes
    for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        nodesAndCoordinates[&(elmDof.GetNode(i))] =
            Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      }
    }
  }

  //! Adds a 2D unit mesh boundary
  void AddMesh2DBoundary() {
    for (ElementCollectionFem &elmColl : mesh.Elements) {
      ElementFem &cElm = elmColl.CoordinateElement();
      ElementFem &dElm = elmColl.DofElement(dof);
      Eigen::MatrixXd coords(dElm.GetNumNodes(), 2);
      for (int i = 0; i < dElm.GetNumNodes(); i++) {
        coords.row(i) =
            Interpolate(cElm, dElm.Interpolation().GetLocalCoords(i));
      }
    }
  }

  //! Adds a 3D unit mesh boundary
  void AddMesh3DBoundary() {
    for (ElementCollectionFem &elmColl : mesh.Elements) {
      ElementFem &cElm = elmColl.CoordinateElement();
      ElementFem &dElm = elmColl.DofElement(dof);
      Eigen::MatrixXd coords(dElm.GetNumNodes(), 3);
      for (int i = 0; i < dElm.GetNumNodes(); i++) {
        coords.row(i) =
            Interpolate(cElm, dElm.Interpolation().GetLocalCoords(i));
      }
    }
  }

  //! Adds a 3D mesh with quad elements in range [0,1] * [0,1] * [0,1]
  void AddMesh3D(int numElm) {
    mesh = NuTo::UnitMeshFem::CreateBricks(numElm, numElm, numElm);
  }

  //! Adds (linear line) interpolation of the single scalar dof type
  void AddFEMApproximation1D() {
    const InterpolationSimple &interpolation =
        //        mesh.CreateInterpolation(InterpolationTrussLinear());
        mesh.CreateInterpolation(InterpolationTrussLobatto(order));
    AddDofInterpolation(&mesh, dof, interpolation);
  }

  //! Adds (linear quads) interpolation of the single scalar dof type
  //! Interpolation must match created elements
  void AddFEMApproximation2D() {
    const InterpolationSimple &interpolation =
        mesh.CreateInterpolation(InterpolationQuadLinear());
    AddDofInterpolation(&mesh, dof, interpolation);
  }

  //! Adds (linear bricks) interpolation of the single scalar dof type
  //! Interpolation must match created elements (at the moment there are no
  //! bricks...)
  void AddFEMApproximation3D() {
    const InterpolationSimple &interpolation =
        // mesh.CreateInterpolation(InterpolationBrickLinear());
        mesh.CreateInterpolation(InterpolationBrickLobatto(2));
    AddDofInterpolation(&mesh, dof, interpolation);
  }

  //! Fix left and right node
  void SetDirichletBoundary1D() {
    Eigen::VectorXd coord(1);
    coord(0) = 0.;
    NodeSimple &left = mesh.NodeAtCoordinate(coord, dof);

    coord(0) = 1.;
    NodeSimple &right = mesh.NodeAtCoordinate(coord, dof);

    // Fixed nonzero values
    // constraints.Add(dof, Constraint::Component(left, {eDirection::X}, 0.1));
    // constraints.Add(dof, Constraint::Component(right, {eDirection::X}, 0.3));

    // Time dependent constraint

    //    constraints.Add(
    //        dof, Constraint::Component(left, {eDirection::X},
    //                                   [](double t) { return sin(2 * M_PI *
    //                                   t); }));
    constraints.Add(
        dof, Constraint::Component(
                 left, {eDirection::X},
                 [&](double t) { return smearedStepFunction(t, riseTime); }));
    constraints.Add(dof, Constraint::Component(right, {eDirection::X}, 0.0));
  }

  bool CoordinatesOnUnitMeshBoundary(Eigen::VectorXd coord) {
    for (int i = 0; i < coord.size(); i++) {
      if ((coord[i] == 0.) || (coord[i] == 1.))
        return true;
    }
    return false;
  }

  //! Set values at the Dirichlet boundary
  void SetDirichletBoundary() {

    std::map<NodeSimple *, Eigen::VectorXd> boundaryNodesAndCoordinates;

    Eigen::VectorXd coord;

    // Get Coordinates of nodes
    for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        coord = Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        boundaryNodesAndCoordinates[&(elmDof.GetNode(i))] = coord;
      }
    }

    // auto gFunc = [](Eigen::VectorXd x) { return (x[0] * x[0] - x[1] * x[1]);
    // };
    auto gFunc = [](Eigen::VectorXd x) { return 0; };

    for (auto &p : boundaryNodesAndCoordinates) {
      if (CoordinatesOnUnitMeshBoundary(p.second)) {
        constraints.Add(dof, Constraint::Value(*(p.first), gFunc(p.second)));
      }
    }
  }

  void NumberDofs() {
    dofInfo = DofNumbering::Build(mesh.NodesTotal(dof), dof, constraints);
  }

  void SetupConstraintMatrix() {
    cmat =
        constraints.BuildConstraintMatrix(dof, dofInfo.numIndependentDofs[dof]);
  }

  void SetupAssembler() {
    asmbl =
        SimpleAssembler(dofInfo.numIndependentDofs, dofInfo.numDependentDofs);
  }

  void AddIntegrationCells1D() {
    integrationType1D =
        IntegrationTypeTensorProduct<1>(order + 1, eIntegrationMethod::LOBATTO);

    int cellId = 0;
    for (ElementCollection &element : mesh.Elements) {
      integrationCells.push_back(
          new Cell(element, integrationType1D, cellId++));
    }
  }

  void AddIntegrationCells2D() {
    integrationType2D =
        IntegrationTypeTensorProduct<2>(2, eIntegrationMethod::LOBATTO);

    int cellId = 0;
    for (ElementCollection &element : mesh.Elements) {
      integrationCells.push_back(
          new Cell(element, integrationType2D, cellId++));
    }
  }

  void AddIntegrationCells3D() {
    integrationType3D =
        IntegrationTypeTensorProduct<3>(3, eIntegrationMethod::LOBATTO);

    int cellId = 0;
    for (ElementCollection &element : mesh.Elements) {
      integrationCells.push_back(
          new Cell(element, integrationType3D, cellId++));
    }
  }

  void AddCellGroup() {
    for (CellInterface &c : integrationCells) {
      allCells.Add(c);
    }
  }

  void ComputeLumpedMassMatrix() {

    lumpedMassMx.J[dof].resize(dofInfo.numIndependentDofs[dof]);
    lumpedMassMx.K[dof].resize(dofInfo.numDependentDofs[dof]);

    lumpedMassMx.J[dof].setZero();
    lumpedMassMx.K[dof].setZero();

    auto f = [&](const CellData /*&cellData*/, const CellIpData &cellIpData) {

      Eigen::MatrixXd N = cellIpData.GetNMatrix(dof);
      DofMatrix<double> massLocal;

      massLocal(dof, dof) = N.transpose() * N * rho;
      return massLocal;
    };

    auto totalMassFunction = [&](const CellData &cellData,
                                 const CellIpData &cellIpData) { return rho; };

    for (NuTo::CellInterface &cell : allCells) {

      // Compute local mass matrix
      DofMatrix<double> cellHessian = cell.Integrate(f);
      // Now do mass lumping (sum all dimensions and later divide)
      const double totalMass = cell.Integrate(totalMassFunction);
      double diagonalMass = cellHessian(dof, dof).diagonal().sum();
      double dim = 1;
      diagonalMass /= dim;
      double scaleFactor = totalMass / diagonalMass;
      // now scale all components
      Eigen::MatrixXd helper =
          (cellHessian(dof, dof).diagonal() * scaleFactor).asDiagonal();
      cellHessian(dof, dof) = helper;
      Eigen::VectorXi numberingdof = cell.DofNumbering(dof);
      const Eigen::MatrixXd &cellHessianDof = cellHessian(dof, dof);

      for (int i = 0; i < numberingdof.rows(); ++i) {
        const int globalDofNumberI = numberingdof[i];
        const double globalDofValue = cellHessianDof(i, i);

        const bool activeI = globalDofNumberI < dofInfo.numIndependentDofs[dof];

        if (activeI) {
          lumpedMassMx.J[dof].coeffRef(globalDofNumberI) += globalDofValue;
        } else {
          lumpedMassMx.K[dof].coeffRef(globalDofNumberI -
                                       dofInfo.numIndependentDofs[dof]) +=
              globalDofValue;
        } // argh. any better ideas?
      }
    }
  }

  void ComputeMassMatrix() {
    auto massF = [&](const CellData &cellData, const CellIpData &cellIpData) {

      Eigen::MatrixXd N = cellIpData.GetNMatrix(dof);
      DofMatrix<double> massLocal;

      massLocal(dof, dof) = N.transpose() * N;
      return massLocal;
    };
    massMx = asmbl.BuildMatrix(allCells, {dof}, massF);
  }

  void ComputeStiffnessMatrix() {
    auto stiffnessF = [&](const CellData &cellData,
                          const CellIpData &cellIpData) {

      Eigen::MatrixXd B = cellIpData.GetBMatrixGradient(dof);
      DofMatrix<double> stiffnessLocal;

      // **********************************
      //        Susceptibility
      // **********************************
      double D = 1.; // will be a (Dim,Dim) matrix in hifger dimensions

      stiffnessLocal(dof, dof) = B.transpose() * D * B;
      return stiffnessLocal;
    };
    stiffnessMx = asmbl.BuildMatrix(allCells, {dof}, stiffnessF);
  }

  void ComputeLoadVector(double t) {
    auto loadF = [&](const CellData &cellData, const CellIpData &cellIpData) {

      Eigen::MatrixXd N = cellIpData.GetNMatrix(dof);
      DofVector<double> loadLocal;

      // **********************************
      //        LoadFunction
      // **********************************
      Eigen::VectorXd coords = cellIpData.GlobalCoordinates();
      double x = coords(0);
      // double f = x * (1. - x);
      double f = 0.;

      loadLocal[dof] = N.transpose() * f;
      return loadLocal;
    };
    loadVector = asmbl.BuildVector(allCells, {dof}, loadF);
  }

  void ComputeModifiedStiffnessMatrix() {
    auto kJJ = stiffnessMx.JJ(dof, dof);
    auto kJK = stiffnessMx.JK(dof, dof);
    auto kKJ = stiffnessMx.KJ(dof, dof);
    auto kKK = stiffnessMx.KK(dof, dof);

    stiffnessMxMod(dof, dof) = kJJ - cmat.transpose() * kKJ - kJK * cmat +
                               cmat.transpose() * kKK * cmat;
  }

  void ComputeModifiedMassMatrix() {
    auto mJJ = massMx.JJ(dof, dof);
    auto mJK = massMx.JK(dof, dof);
    auto mKJ = massMx.KJ(dof, dof);
    auto mKK = massMx.KK(dof, dof);

    massMxMod(dof, dof) = mJJ - cmat.transpose() * mKJ - mJK * cmat +
                          cmat.transpose() * mKK * cmat;
  }

  void ComputeModifiedLumpedMassMatrix() {
    auto mJJ = lumpedMassMx.J[dof];
    auto mKK = lumpedMassMx.K[dof];

    Eigen::MatrixXd tmp = cmat.transpose() * mKK.asDiagonal() * cmat;

    lumpedMassMxMod[dof] = mJJ + tmp.diagonal();
  }

  //! (Compute necessary time derivatives by forward differences with timestep
  //! dt (test this!) )
  void ComputeModifiedLoadVector(double t, double dt) {
    auto kJK = stiffnessMx.JK(dof, dof);
    auto kKK = stiffnessMx.KK(dof, dof);

    auto mJK = massMx.JK(dof, dof);
    auto mKK = massMx.KK(dof, dof);

    // Only for time dependent loads
    // ComputeLoadVector(t);
    auto fJ = loadVector.J[dof];
    auto fK = loadVector.K[dof];
    auto b = constraints.GetRhs(dof, t);
    // auto bDot = (constraints.GetRhs(dof, t) + constraints.GetRhs(dof, t +
    // dt)) / dt;
    auto bDDot = (constraints.GetRhs(dof, t) -
                  2. * constraints.GetRhs(dof, t + 0.5 * dt) +
                  constraints.GetRhs(dof, t + dt)) /
                 dt;
    loadVectorMod[dof] = fJ - cmat.transpose() * fK;
    loadVectorMod[dof] -= (kJK * b - cmat.transpose() * kKK * b);
    loadVectorMod[dof] -= (mJK * bDDot - cmat.transpose() * mKK * bDDot);
  }

  void SolvePoisson() {
    Eigen::SparseMatrix<double> A = stiffnessMxMod(dof, dof);
    Eigen::VectorXd b = loadVectorMod[dof];
    Eigen::VectorXd x;

    // Compute Independent Dofs
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
      throw Exception("decomposition failed");
    }
    x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
      throw Exception("decomposition failed");
    }
    // Compute Dependent Dofs
    Eigen::VectorXd y = -cmat * x + constraints.GetRhs(dof, 0);

    // StoreResult
    femResult.resize(x.size() + y.size());
    femResult.head(x.size()) = x;
    femResult.tail(y.size()) = y;
  }

  void SolveWave1D(double h, int numSteps, bool useFullMass) {
    NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
    double t = 0.;

    ExtractNodeValsToFEMResult();
    Eigen::VectorXd w0 = femResult.head(dofInfo.numIndependentDofs[dof]);
    Eigen::VectorXd v0 = w0 * 0.;

    auto state = std::make_pair(w0, v0);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    if (useFullMass) {
      solver.compute(massMxMod(dof, dof));
      if (solver.info() != Eigen::Success) {
        throw Exception("decomposition failed");
      }
    }

    auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
      ComputeModifiedLoadVector(t, 0.01);
      if (useFullMass) {
        d2wdt2 =
            -solver.solve(stiffnessMxMod(dof, dof) * w - loadVectorMod[dof]);
        if (solver.info() != Eigen::Success) {
          throw Exception("solve failed");
        }
      } else {
        auto tmp = (stiffnessMxMod(dof, dof) * w - loadVectorMod[dof]);
        d2wdt2 = (tmp.array() / lumpedMassMx.J[dof].array()).matrix();
      }
    };

    // print out initial value
    std::cout << 0 << std::endl;
    MergeResult();
    VisualizeResult1D("Wave1Dtest_0");

    for (int i = 0; i < numSteps; i++) {
      t = i * h;
      state = ti.DoStep(eq, state.first, state.second, t, h);
      femResult.head(dofInfo.numIndependentDofs[dof]) = state.first;
      // Compute Dependent Dofs
      femResult.tail(dofInfo.numDependentDofs[dof]) =
          -cmat * state.first + constraints.GetRhs(dof, t);
      MergeResult();
      std::cout << i + 1 << std::endl;
      if ((i * 100) % numSteps == 0) {
        VisualizeResult1D("Wave1Dtest_" + std::to_string(i + 1));

        MergeExactSolution(t);
        VisualizeResult1D("Wave1Dexact_" + std::to_string(i + 1));
      }
    }
  }

  void SolveWave2D() {
    NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
    double h = 0.005;
    double t = 0.;
    double numSteps = 200;

    ExtractNodeValsToFEMResult();
    Eigen::VectorXd w0 = femResult.head(dofInfo.numIndependentDofs[dof]);
    Eigen::VectorXd v0 = w0 * 0.;

    auto state = std::make_pair(w0, v0);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(massMxMod(dof, dof));
    if (solver.info() != Eigen::Success) {
      throw Exception("decomposition failed");
    }

    auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
      ComputeModifiedLoadVector(t, 0.01);
      d2wdt2 = -solver.solve(stiffnessMxMod(dof, dof) * w) + loadVectorMod[dof];
      if (solver.info() != Eigen::Success) {
        throw Exception("solve failed");
      }
    };

    // print out initial value
    std::cout << 0 << std::endl;
    MergeResult();
    VisualizeResult2D("Wave2Dtest_0");

    for (int i = 0; i < numSteps; i++) {
      t = i * h;
      state = ti.DoStep(eq, state.first, state.second, t, h);
      femResult.head(dofInfo.numIndependentDofs[dof]) = state.first;
      // Compute Dependent Dofs
      femResult.tail(dofInfo.numDependentDofs[dof]) =
          -cmat * state.first + constraints.GetRhs(dof, 0);
      MergeResult();
      std::cout << i + 1 << std::endl;
      VisualizeResult2D("Wave2Dtest_" + std::to_string(i + 1));
    }
  }

  void SolveWave3D() {
    NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
    double h = 0.005;
    double t = 0.;
    double numSteps = 200;

    ExtractNodeValsToFEMResult();
    Eigen::VectorXd w0 = femResult.head(dofInfo.numIndependentDofs[dof]);
    Eigen::VectorXd v0 = w0 * 0.;

    auto state = std::make_pair(w0, v0);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(massMxMod(dof, dof));
    if (solver.info() != Eigen::Success) {
      throw Exception("decomposition failed");
    }

    auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
      ComputeModifiedLoadVector(t, 0.01);
      d2wdt2 = -solver.solve(stiffnessMxMod(dof, dof) * w) + loadVectorMod[dof];
      if (solver.info() != Eigen::Success) {
        throw Exception("solve failed");
      }
    };

    // print out initial value
    std::cout << 0 << std::endl;
    MergeResult();
    VisualizeResult3D("Wave3Dtest_0");

    for (int i = 0; i < numSteps; i++) {
      t = i * h;
      state = ti.DoStep(eq, state.first, state.second, t, h);
      femResult.head(dofInfo.numIndependentDofs[dof]) = state.first;
      // Compute Dependent Dofs
      femResult.tail(dofInfo.numDependentDofs[dof]) =
          -cmat * state.first + constraints.GetRhs(dof, 0);
      MergeResult();
      std::cout << i + 1 << std::endl;
      VisualizeResult3D("Wave3Dtest_" + std::to_string(i + 1));
    }
  }

  void SolveDiffusion1D() {
    ImplicitEuler<Eigen::VectorXd, Eigen::MatrixXd> ti;
    double h = 0.005;
    double t = 0.;
    double numSteps = 200;

    ExtractNodeValsToFEMResult();
    Eigen::VectorXd w = femResult.head(dofInfo.numIndependentDofs[dof]);

    Eigen::MatrixXd massFull = massMxMod(dof, dof);
    systemMx = massFull.inverse() * stiffnessMxMod(dof, dof);

    auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &dwdt, double t) {
      dwdt = -systemMx * w;
    };
    auto jac = [&](const Eigen::VectorXd &w, Eigen::MatrixXd &dwdt, double t) {
      dwdt = -systemMx;
    };

    // print out initial value
    std::cout << w << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < numSteps; i++) {
      t = i * h;
      w = ti.DoStep(eq, jac, w, w, t, h);
      std::cout << w << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
    }
  }

  void SolveDiffusion2D() {
    ImplicitEuler<Eigen::VectorXd, Eigen::MatrixXd> ti;
    double h = 0.005;
    double t = 0.;
    double numSteps = 200;

    ExtractNodeValsToFEMResult();
    Eigen::VectorXd w = femResult.head(dofInfo.numIndependentDofs[dof]);

    Eigen::MatrixXd massFull = massMxMod(dof, dof);
    systemMx = massFull.inverse() * stiffnessMxMod(dof, dof);

    auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &dwdt, double t) {
      dwdt = -systemMx * w;
    };
    auto jac = [&](const Eigen::VectorXd &w, Eigen::MatrixXd &dwdt, double t) {
      dwdt = -systemMx;
    };

    // print out initial value
    std::cout << 0 << std::endl;
    MergeResult();
    VisualizeResult2D("Diffusion2Dtest_0");

    for (int i = 0; i < numSteps; i++) {
      t = i * h;
      w = ti.DoStep(eq, jac, w, w, t, h);
      femResult.head(dofInfo.numIndependentDofs[dof]) = w;
      // Compute Dependent Dofs
      femResult.tail(dofInfo.numDependentDofs[dof]) =
          -cmat * w + constraints.GetRhs(dof, 0);
      MergeResult();
      std::cout << i + 1 << std::endl;
      VisualizeResult2D("Diffusion2Dtest_" + std::to_string(i + 1));
    }
  }

  void SolveDiffusion3D() {
    ImplicitEuler<Eigen::VectorXd, Eigen::MatrixXd> ti;
    double h = 0.005;
    double t = 0.;
    double numSteps = 4;

    ExtractNodeValsToFEMResult();
    Eigen::VectorXd w = femResult.head(dofInfo.numIndependentDofs[dof]);

    Eigen::MatrixXd massFull = massMxMod(dof, dof);
    systemMx = massFull.inverse() * stiffnessMxMod(dof, dof);

    auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &dwdt, double t) {
      dwdt = -systemMx * w;
    };
    auto jac = [&](const Eigen::VectorXd &w, Eigen::MatrixXd &dwdt, double t) {
      dwdt = -systemMx;
    };

    // print out initial value
    std::cout << 0 << std::endl;
    MergeResult();
    VisualizeResult3D("Diffusion3Dtest_0");

    for (int i = 0; i < numSteps; i++) {
      t = i * h;
      w = ti.DoStep(eq, jac, w, w, t, h);
      femResult.head(dofInfo.numIndependentDofs[dof]) = w;
      // Compute Dependent Dofs
      femResult.tail(dofInfo.numDependentDofs[dof]) =
          -cmat * w + constraints.GetRhs(dof, 0);
      MergeResult();
      std::cout << i + 1 << std::endl;
      VisualizeResult2D("Diffusion3Dtest_" + std::to_string(i + 1));
    }
  }

  void SolveHelmholtz() {
    Eigen::MatrixXd M = massMxMod(dof, dof);
    Eigen::MatrixXd K = stiffnessMxMod(dof, dof);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(M, K);

    auto vals = solver.eigenvalues();
    //        for (size_t i = 0; i < vals.size(); i++)
    //        {
    //            std::cout << vals[i] << std::endl;
    //        }

    // plot the first few modes
    auto vecs = solver.eigenvectors();
    int nModes = 10;
    for (int i = 0; i < nModes; i++) {
      std::cout << vecs.col(vecs.cols() - 1 - i) << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
    }
  }

  void MergeResult() {
    for (auto &node : mesh.NodesTotal(dof)) {
      int dofNr = node.GetDofNumber(0);
      node.SetValue(0, femResult[dofNr]);
    }
  }

  void MergeExactSolution(double t) {
    for (auto &node : mesh.NodesTotal(dof)) {
      node.SetValue(0, ExactResultWave1D(nodesAndCoordinates.at(&node)[0], t));
    }
  }

  void ExtractNodeValsToFEMResult() {
    femResult.resize(dofInfo.numDependentDofs[dof] +
                     dofInfo.numIndependentDofs[dof]);
    for (auto &node : mesh.NodesTotal(dof)) {
      int dofNr = node.GetDofNumber(0);
      femResult[dofNr] = node.GetValues()(0);
    }
  }

  void SetInitialCondition1D() {
    auto func = [](double x) {
      if ((1. / 3 < x) && (x < 2. / 3)) {
        return (0.5 * (1. - cos(2. * M_PI * 3 * (x - 1. / 3))));
      } else {
        return 0.;
      }
    };

    for (ElementCollectionFem &elmColl : mesh.Elements) {
      ElementFem &elm0 = elmColl.CoordinateElement();
      ElementFem &elm1 = elmColl.DofElement(dof);
      for (int i = 0; i < elm1.Interpolation().GetNumNodes(); i++) {
        Eigen::VectorXd y = elm1.Interpolation().GetLocalCoords(i);
        double x = Interpolate(elm0, y)(0);
        auto &node = elm1.GetNode(i);
        node.SetValue(0, func(x));
      }
    }
  }

  void SetInitialCondition2D() {
    auto func = [](double x) {
      if ((1. / 3 < x) && (x < 2. / 3)) {
        return (0.5 * (1. - cos(2. * M_PI * 3 * (x - 1. / 3))));
      } else {
        return 0.;
      }
    };

    for (ElementCollectionFem &elmColl : mesh.Elements) {
      ElementFem &elm0 = elmColl.CoordinateElement();
      ElementFem &elm1 = elmColl.DofElement(dof);
      for (int i = 0; i < elm1.Interpolation().GetNumNodes(); i++) {
        Eigen::VectorXd xi = elm1.Interpolation().GetLocalCoords(i);
        Eigen::Vector2d coords = Interpolate(elm0, xi);
        auto &node = elm1.GetNode(i);
        node.SetValue(0, func(coords[0]) * func(coords[1]));
      }
    }
  }

  void SetInitialCondition3D() {
    auto func = [](double x) {
      if ((1. / 3 < x) && (x < 2. / 3)) {
        return (0.5 * (1. - cos(2. * M_PI * 3 * (x - 1. / 3))));
      } else {
        return 0.;
      }
    };

    for (ElementCollectionFem &elmColl : mesh.Elements) {
      ElementFem &elm0 = elmColl.CoordinateElement();
      ElementFem &elm1 = elmColl.DofElement(dof);
      for (int i = 0; i < elm1.Interpolation().GetNumNodes(); i++) {
        Eigen::VectorXd xi = elm1.Interpolation().GetLocalCoords(i);
        Eigen::Vector3d coords = Interpolate(elm0, xi);
        auto &node = elm1.GetNode(i);
        node.SetValue(0, func(coords[0]) * func(coords[1]) * func(coords[2]));
      }
    }
  }

  //  void VisualizeResult1D(std::string filename) {
  //    NuTo::Visualize::UnstructuredGrid grid;
  //    grid.DefinePointData(dof.GetName());

  //    for (ElementCollectionFem &elmColl : mesh.Elements) {
  //      std::vector<int> pointIds;
  //      ElementFem &elm0 = elmColl.CoordinateElement();
  //      ElementFem &elm1 = elmColl.DofElement(dof);
  //      for (int i = 0; i < elm1.Interpolation().GetNumNodes(); i++) {
  //        Eigen::VectorXd y = elm1.Interpolation().GetLocalCoords(i);
  //        Eigen::VectorXd x = Interpolate(elm0, y);
  //        Eigen::Vector3d coords;
  //        coords << x(0), 0., 0.;
  //        int pId = grid.AddPoint(coords);
  //        pointIds.push_back(pId);
  //        grid.SetPointData(pId, dof.GetName(),
  //        elm1.GetNode(i).GetValues());
  //      }
  //      for (int i = 0; i < elm1.Interpolation().GetNumNodes() - 1; i++) {
  //        grid.AddCell({pointIds[i], pointIds[i + 1]},
  //        NuTo::eCellTypes::LINE);
  //      }
  //    }
  //    NuTo::Visualize::XMLWriter::Export(filename + ".vtu", grid, false);
  //  }

  void VisualizeResult1D(std::string filename) {

    NuTo::Visualize::Visualizer<NuTo::Visualize::VoronoiHandler> visualize(
        allCells, NuTo::Visualize::VoronoiGeometryLine(order * 20));
    visualize.DofValues(dof);
    visualize.WriteVtuFile(filename + ".vtu");
  }

  double ExactResultWave1D(double x, double t) {
    return smearedStepFunction(t - x, riseTime);
  }

  void VisualizeResult2D(std::string filename) {

    //    NuTo::Visualize::Visualizer<NuTo::Visualize::AverageHandler>
    //    visualize(
    //        allCells, NuTo::Visualize::AverageGeometryQuad());
    NuTo::Visualize::Visualizer<NuTo::Visualize::VoronoiHandler> visualize(
        allCells, NuTo::Visualize::VoronoiGeometryQuad(5));
    visualize.DofValues(dof);
    visualize.WriteVtuFile(filename + ".vtu");
  }

  void VisualizeResult3D(std::string filename) {

    //    NuTo::Visualize::Visualizer<NuTo::Visualize::AverageHandler>
    //    visualize(
    //        allCells, NuTo::Visualize::AverageGeometryBrick());
    NuTo::Visualize::Visualizer<NuTo::Visualize::VoronoiHandler> visualize(
        allCells, NuTo::Visualize::VoronoiGeometryBrick(2));
    visualize.DofValues(dof);
    visualize.WriteVtuFile(filename + ".vtu");
  }

  void ApplyNeumannBoundary() {
    // need a call to the assembler for that
    // need boundary elements
    // then add it to the "gradient" / load vector
  }

  void Initialize() {
    AddMesh1D(numElm);
    //    AddMesh2D(numElm);
    //    AddMesh3D(numElm); // not working, unit::mesh::createBricks would be
    //    nice

    AddFEMApproximation1D();
    // AddFEMApproximation2D();
    // AddFEMApproximation3D();

    AddMesh1DBoundary();
    // AddMesh2DBoundary();
    // AddMesh3DBoundary();

    FillNodesAndCoordinatesMap();

    SetDirichletBoundary1D();
    // SetDirichletBoundary(); // this definitely needs attention !!!!

    NumberDofs();            // --> Assembler
    SetupConstraintMatrix(); // --> Assembler
    SetupAssembler();

    AddIntegrationCells1D();
    // AddIntegrationCells2D();
    // AddIntegrationCells3D();

    AddCellGroup();

    ApplyNeumannBoundary();

    ComputeMassMatrix();
    ComputeLumpedMassMatrix();
    ComputeStiffnessMatrix();
    ComputeLoadVector(0.);
    ComputeModifiedMassMatrix();       // --> ModelProblemAdapter
    ComputeModifiedLumpedMassMatrix(); // --> ModelProblemAdapter
    ComputeModifiedStiffnessMatrix();  // --> ModelProblemAdapter
    ComputeModifiedLoadVector(0., 0.01);
  }
};

int main(int argc, char *argv[]) {
  //    std::cout << "Test Poisson type equations: \n";
  //    std::cout << "1 Laplace equation \n";
  //    std::cout << "2 Poisson equation \n";
  //    std::cout << "3 Diffusion/Heat equation \n";
  //    std::cout << "4 Wave equation \n";
  //    std::cout << "5 Helmholtz equation \n";
  //    std::cout << "6 Fisher equation \n";
  //    std::cout << "7 Burgers equation \n";
  //    std::cout << std::endl;

  int order = 35;
  int numElements = 1;
  PoissonTypeProblem example1(order, numElements);
  //  example1.Initialize();

  // ************ Laplace / Poisson  ****************

  //  example1.SolvePoisson();
  //  example1.MergeResult();
  //  example1.VisualizeResult1D("Poisson1D");
  // For Voronoi Lobatto nodes are missing
  //  example1.VisualizeResult2D("Poisson2Dtest");
  //  example1.VisualizeResult3D("Poisson3Dtest");

  // ************ Wave equation *********************

  // example1.SetInitialCondition1D();
  // example1.SetInitialCondition2D();
  // example1.SetInitialCondition3D();

  //  std::cout << example1.massMx.JJ(example1.dof, example1.dof) << std::endl;
  //  std::cout << Eigen::MatrixXd(
  //                   example1.lumpedMassMx.J[example1.dof].asDiagonal())
  //            << std::endl;

  //  std::cout << example1.massMxMod(example1.dof, example1.dof) << std::endl;
  //  std::cout << Eigen::MatrixXd(
  //                   example1.lumpedMassMxMod[example1.dof].asDiagonal())
  //            << std::endl;

  // example1.SolveWave1D(0.005, 200, false);
  // example1.SolveWave1D(0.005, 200, true);
  // example1.SolveWave2D();
  // example1.SolveWave3D();
  // example1.VisualizeResult2D("Wave2Dtest");

  // ************ Diffusion / Heat equation *********

  //    example1.SetInitialCondition();
  // example1.SetInitialCondition2D();
  // example1.SolveDiffusion1D();
  // example1.SolveDiffusion2D();

  // ************ Helmholtz equation ****************

  //    example1.SolveHelmholtz();

  // ************ Fisher equation ****************
  // ************ Burgers equation ****************
}
