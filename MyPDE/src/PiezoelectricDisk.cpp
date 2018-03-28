#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/math/EigenCompanion.h"
#include "nuto/mechanics/tools/CellStorage.h"
#include "nuto/mechanics/tools/NodalValueMerger.h"

#include "nuto/mechanics/dofs/DofMatrixSparseConvertEigen.h"
#include "nuto/mechanics/dofs/DofVectorConvertEigen.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"
#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/Piezoelectricity.h"
#include "MaterialConstants.h"

#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {
  // ***************************
  //    Some parameters
  // ***************************

  int order = 2;

  // ***************************
  //      Import a mesh
  // ***************************

  // MeshGmsh gmsh("disk02.msh");
  // MeshGmsh gmsh("diskRegular_small.msh");
  // MeshGmsh gmsh("diskRegular_middle.msh");
  // MeshGmsh gmsh("cube5.msh");
  MeshGmsh gmsh("cube1.msh");
  // MeshGmsh gmsh("diskRegular_smallLinear.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto outerBoundary = gmsh.GetPhysicalGroup("OuterBoundary");
  auto bottomOuterRing = gmsh.GetPhysicalGroup("BottomOuterRing");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  auto boundary = Unite(Unite(top, bottom), outerBoundary);
  auto dirichletBoundaryElec = boundary;
  auto dirichletBoundaryMech = boundary;
  auto neumannBoundaryElec = Difference(boundary, dirichletBoundaryElec);
  auto neumannBoundaryMech = Difference(boundary, dirichletBoundaryMech);

  // ***************************
  //      Set up equations
  // ***************************
  Eigen::Matrix3d permittivity = NCE51::permittivity_S;
  Eigen::MatrixXd stiffness = NCE51::stiffness_E;
  Eigen::MatrixXd piezo = NCE51::piezo_e;

  Integrands::Piezoelectricity pde(stiffness, piezo, permittivity);

  auto &ipol = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, pde.dofU, domain, ipol);
  AddDofInterpolation(&mesh, pde.dofV, domain, ipol);

  auto &ipolBoundary =
      mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, pde.dofU, boundary, ipolBoundary);
  AddDofInterpolation(&mesh, pde.dofV, boundary, ipolBoundary);

  auto &ipolRing = mesh.CreateInterpolation(InterpolationTrussLobatto(order));
  AddDofInterpolation(&mesh, pde.dofU, bottomOuterRing, ipolRing);
  AddDofInterpolation(&mesh, pde.dofV, bottomOuterRing, ipolRing);

  // ***************************
  //   Set up a solution
  // ***************************

  // prescribed homogeneous fields:
  Eigen::VectorXd homStrainVoigt(6);
  homStrainVoigt << 0., 0., 0., 0., 0., 0.;

  Eigen::VectorXd homEField(3);
  homEField << 0.0, 0.0, 3.14;

  // integration constants
  double groundPotential = 0.;
  Eigen::Vector3d originDisplacements(0., 0., 0.);
  Eigen::Vector3d rotVec(0., 0., 0.);

  // computed quantities
  Eigen::Matrix3d homStrain;
  homStrain(0, 0) = homStrainVoigt[0];
  homStrain(1, 1) = homStrainVoigt[1];
  homStrain(2, 2) = homStrainVoigt[2];

  homStrain(1, 2) = homStrainVoigt[3] / 2.;
  homStrain(2, 0) = homStrainVoigt[4] / 2.;
  homStrain(0, 1) = homStrainVoigt[5] / 2.;

  homStrain(2, 1) = homStrainVoigt[3] / 2.;
  homStrain(0, 2) = homStrainVoigt[4] / 2.;
  homStrain(1, 0) = homStrainVoigt[5] / 2.;

  /*
   * Ax = a cross X
   *
   * |  0   A12 A13 |     ay Xz - az Xy   |     - az   ay |
   * |-A12  0   A23 | x = az Xx - ax Xz = |  az   0  - ax | X
   * |-A13 -A23 0   |     ax Xy - ay Xx   |- ay  ax       |
   *
  */
  Eigen::Matrix3d rotA;
  rotA << 0., -rotVec[2], rotVec[1], //
      rotVec[2], 0., -rotVec[0],     //
      -rotVec[1], rotVec[0], 0.;     //

  auto potentialFunc = [&](Eigen::Vector3d x) {
    return -homEField.dot(x) + groundPotential;
  };

  auto displacementFunc = [&](Eigen::Vector3d x) {
    return (homStrain + rotA) * x + originDisplacements;
  };

  Eigen::VectorXd homStressVoigt =
      stiffness * homStrainVoigt - piezo.transpose() * homEField;

  Eigen::Matrix3d homStress;
  homStress(0, 0) = homStressVoigt[0];
  homStress(1, 1) = homStressVoigt[1];
  homStress(2, 2) = homStressVoigt[2];

  homStress(1, 2) = homStressVoigt[3];
  homStress(2, 0) = homStressVoigt[4];
  homStress(0, 1) = homStressVoigt[5];

  homStress(2, 1) = homStressVoigt[3];
  homStress(0, 2) = homStressVoigt[4];
  homStress(1, 0) = homStressVoigt[5];

  Eigen::VectorXd homDField = piezo * homStrainVoigt + permittivity * homEField;

  // ***************************
  //    Boundary conditions
  // ***************************

  Group<NodeSimple> dirichletBoundaryNodesMech;
  for (ElementCollectionFem &elmColl : dirichletBoundaryMech) {
    NuTo::ElementFem &elmDofU = elmColl.DofElement(pde.dofU);
    for (int i = 0; i < elmDofU.GetNumNodes(); i++) {
      dirichletBoundaryNodesMech.Add(elmDofU.GetNode(i));
    }
  }

  Group<NodeSimple> dirichletBoundaryNodesElec;
  for (ElementCollectionFem &elmColl : dirichletBoundaryElec) {
    NuTo::ElementFem &elmDofV = elmColl.DofElement(pde.dofV);
    for (int i = 0; i < elmDofV.GetNumNodes(); i++) {
      dirichletBoundaryNodesElec.Add(elmDofV.GetNode(i));
    }
  }

  Constraint::Constraints constraints;

  constraints.Add(pde.dofV,
                  Constraint::SetDirichletBoundaryNodes(
                      pde.dofV, dirichletBoundaryElec, potentialFunc));
  constraints.Add(pde.dofU,
                  Constraint::SetDirichletBoundaryNodes(
                      pde.dofU, dirichletBoundaryMech, displacementFunc));

  // ******************************
  //    Set up assembler
  // ******************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(pde.dofU), pde.dofU, constraints);

  dofInfo.Merge(pde.dofV, DofNumbering::Build(mesh.NodesTotal(pde.dofV),
                                              pde.dofV, constraints));

  DofMatrixSparse<double> cmat;
  for (auto dofI : pde.GetDofs())
    for (auto dofJ : pde.GetDofs()) {
      if (dofI.Id() == dofJ.Id())
        cmat(dofI, dofI) = constraints.BuildConstraintMatrix(
            dofI, dofInfo.numIndependentDofs[dofI]);
      else {
        cmat(dofI, dofJ).setZero();
      }
    }
  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  IntegrationTypeTensorProduct<3> integrationType3D(
      order + 1, eIntegrationMethod::LOBATTO);
  IntegrationTypeTensorProduct<2> integrationType2D(
      order + 1, eIntegrationMethod::LOBATTO);

  // volume cells
  CellStorage volumeCells;
  Group<CellInterface> volumeCellGroup =
      volumeCells.AddCells(domain, integrationType3D);

  // boundary cells
  CellStorage neumannCellsElec;
  Group<CellInterface> neumannCellGroupElec =
      neumannCellsElec.AddCells(neumannBoundaryElec, integrationType2D);

  CellStorage neumannCellsMech;
  Group<CellInterface> neumannCellGroupMech =
      neumannCellsMech.AddCells(neumannBoundaryMech, integrationType2D);

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  GlobalDofMatrixSparse stiffnessMx = asmbl.BuildMatrix(
      volumeCellGroup, pde.GetDofs(),
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  // Compute modified stiffness matrix

  auto kJJ = ToEigen(stiffnessMx.JJ, pde.GetDofs());
  auto kJK = ToEigen(stiffnessMx.JK, pde.GetDofs());
  auto kKJ = ToEigen(stiffnessMx.KJ, pde.GetDofs());
  auto kKK = ToEigen(stiffnessMx.KK, pde.GetDofs());
  auto cmatE = ToEigen(cmat, pde.GetDofs());

  Eigen::SparseMatrix<double> stiffnessMxMod = kJJ;
  stiffnessMxMod -= cmatE.transpose() * kKJ;
  stiffnessMxMod -= kJK * cmatE;
  stiffnessMxMod += cmatE.transpose() * kKK * cmatE;

  // ***********************************
  //    Assemble load vector
  // ***********************************

  GlobalDofVector loadVector;
  for (auto dof : pde.GetDofs()) {
    loadVector.J[dof].setZero(dofInfo.numIndependentDofs[dof]);
    loadVector.K[dof].setZero(dofInfo.numDependentDofs[dof]);
  }

  GlobalDofVector loadVectorU = asmbl.BuildVector(
      neumannCellGroupMech, {pde.dofU}, [&](const CellIpData &cipd) {
        Eigen::VectorXd normal = cipd.GetJacobian().Normal();
        Eigen::VectorXd traction = homStress * normal;
        return pde.NeumannLoadMechanical(cipd, traction);
      });

  GlobalDofVector loadVectorV = asmbl.BuildVector(
      neumannCellGroupElec, {pde.dofV}, [&](const CellIpData &cipd) {
        return pde.NeumannLoadElectrical(
            cipd, homDField.dot(cipd.GetJacobian().Normal()));
      });

  loadVector.J[pde.dofU] += loadVectorU.J[pde.dofU];
  loadVector.K[pde.dofU] += loadVectorU.K[pde.dofU];

  loadVector.J[pde.dofV] += loadVectorV.J[pde.dofV];
  loadVector.K[pde.dofV] += loadVectorV.K[pde.dofV];

  // Compute modified load vector
  Eigen::VectorXd fJ = ToEigen(loadVector.J, pde.GetDofs());
  Eigen::VectorXd fK = ToEigen(loadVector.K, pde.GetDofs());
  Eigen::VectorXd loadVectorMod = fJ - cmatE.transpose() * fK;

  DofVector<double> bRhs;
  for (auto dof : pde.GetDofs())
    bRhs[dof] = constraints.GetRhs(dof, 0.);
  loadVectorMod -=
      (kJK - cmatE.transpose() * kKK) * ToEigen(bRhs, pde.GetDofs());

  // ***********************************
  //    Solve system
  // ***********************************

  int numIndependentDofs = 0;
  int numDependentDofs = 0;
  for (DofType d : pde.GetDofs()) {
    numIndependentDofs += dofInfo.numIndependentDofs.At(d);
    numDependentDofs += dofInfo.numDependentDofs.At(d);
  }

  std::cout << "Num independent dofs: " << numIndependentDofs << "\n";
  std::cout << "Num dependent dofs  : " << numDependentDofs << std::endl;

  Eigen::VectorXd femResultIndependent(numIndependentDofs);
  Eigen::VectorXd femResultDependent(numDependentDofs);

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(stiffnessMxMod);
  if (solver.info() != Eigen::Success) {
    throw Exception("Decomposition failed");
  }
  femResultIndependent = solver.solve(loadVectorMod);
  if (solver.info() != Eigen::Success) {
    throw Exception("Solving failed");
  }

  femResultDependent =
      -cmatE * femResultIndependent + ToEigen(bRhs, pde.GetDofs());

  GlobalDofVector globalSolution = loadVector;
  globalSolution.J *= 0.;
  globalSolution.K *= 0.;
  FromEigen(femResultIndependent, pde.GetDofs(), &globalSolution.J);
  FromEigen(femResultDependent, pde.GetDofs(), &globalSolution.K);

  // ***********************************
  //    Merge
  // ***********************************

  NuTo::NodalValueMerger merger(&mesh);
  merger.Merge(globalSolution, pde.GetDofs());

  GlobalDofVector gradient = asmbl.BuildVector(
      volumeCellGroup, pde.GetDofs(),
      [&](const CellIpData &cipd) { return pde.Gradient(cipd); });

  // ***********************************
  //    Visualize
  // ***********************************

  NuTo::Visualize::Visualizer visualize(
      volumeCellGroup,
      NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryBrick(
          order + 1, Visualize::LOBATTO)));
  visualize.DofValues(pde.dofU);
  visualize.DofValues(pde.dofV);

  visualize.CellData([&](CellIpData cipd) { return pde.Stress(cipd); },
                     "stress");
  visualize.CellData([&](CellIpData cipd) { return pde.DField(cipd); },
                     "DField");

  visualize.CellData([&](CellIpData cipd) { return pde.Strain(cipd); },
                     "strain");
  visualize.CellData([&](CellIpData cipd) { return pde.EField(cipd); },
                     "EField");

  visualize.CellData([&](CellIpData cipd) { return homStressVoigt; },
                     "stressExact");
  visualize.CellData([&](CellIpData cipd) { return homDField; }, "DFieldExact");

  visualize.PointData(
      [&](Eigen::VectorXd x) {
        return Eigen::VectorXd::Constant(1, potentialFunc(x));
      },
      "PotentialExact");
  visualize.PointData(displacementFunc, "DisplacementsExact");

  visualize.WriteVtuFile("Piezoelectricity.vtu");
}
