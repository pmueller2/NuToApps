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

#include "nuto/mechanics/dofs/DofMatrixSparseConvertEigen.h"
#include "nuto/mechanics/dofs/DofVectorConvertEigen.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"
#include "../../NuToHelpers/Piezoelectricity.h"

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

  MeshGmsh gmsh("disk02.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto outerBoundary = gmsh.GetPhysicalGroup("OuterBoundary");
  auto bottomOuterRing = gmsh.GetPhysicalGroup("BottomOuterRing");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  auto boundary = Unite(Unite(top, bottom), outerBoundary);
  auto dirichletBoundary = bottom;
  auto neumannBoundary = top;

  // ***************************
  //      Set up equations
  // ***************************
  Eigen::Matrix3d permittivity;
  permittivity << 1., 0., 0., //
      0., 1., 0.,             //
      0., 0., 1.;             //

  Eigen::MatrixXd stiffness(6, 6);
  stiffness << 1., 0., 0., 0., 0., 0., //
      0., 1., 0., 0., 0., 0.,          //
      0., 0., 1., 0., 0., 0.,          //
      0., 0., 0., 1., 0., 0.,          //
      0., 0., 0., 0., 1., 0.,          //
      0., 0., 0., 0., 0., 1.;          //

  Eigen::MatrixXd piezo(3, 6);
  piezo << 1., 0., 0., 0., 0., 0., //
      0., 1., 0., 0., 0., 0.,      //
      0., 0., 1., 0., 0., 0.;      //

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
  //    Boundary conditions
  // ***************************

  Group<NodeSimple> dirichletBoundaryNodesMech;
  Group<NodeSimple> dirichletBoundaryNodesElec;
  for (ElementCollectionFem &elmColl : dirichletBoundary) {
    NuTo::ElementFem &elmDofU = elmColl.DofElement(pde.dofU);
    NuTo::ElementFem &elmDofV = elmColl.DofElement(pde.dofV);
    for (int i = 0; i < elmDofU.GetNumNodes(); i++) {
      dirichletBoundaryNodesMech.Add(elmColl.DofElement(pde.dofU).GetNode(i));
    }
    for (int i = 0; i < elmDofV.GetNumNodes(); i++) {
      dirichletBoundaryNodesElec.Add(elmColl.DofElement(pde.dofV).GetNode(i));
    }
  }

  Constraint::Constraints constraints;
  constraints.Add(pde.dofU, Constraint::Component(dirichletBoundaryNodesMech,
                                                  {eDirection::Z}, 0.));

  constraints.Add(pde.dofV, Constraint::Value(dirichletBoundaryNodesElec));

  // ******************************
  //    Set up assembler
  // ******************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(pde.dofU), pde.dofU, constraints);

  dofInfo.Merge(pde.dofV, DofNumbering::Build(mesh.NodesTotal(pde.dofV),
                                              pde.dofV, constraints));

  DofMatrixSparse<double> cmat;
  for (auto dofI : {pde.dofU, pde.dofV})
    for (auto dofJ : {pde.dofU, pde.dofV}) {
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
  CellStorage neumannCells;
  Group<CellInterface> neumannCellGroup =
      neumannCells.AddCells(neumannBoundary, integrationType2D);

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  GlobalDofMatrixSparse stiffnessMx = asmbl.BuildMatrix(
      volumeCellGroup, {pde.dofU, pde.dofV},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  // Compute modified stiffness matrix
  std::vector<DofType> dofs{pde.dofU, pde.dofV};

  Eigen::SparseMatrix<double> stiffnessMxMod = ToEigen(stiffnessMx.JJ, dofs);
  auto kJJ = ToEigen(stiffnessMx.JJ, dofs);
  auto kJK = ToEigen(stiffnessMx.JK, dofs);
  auto kKJ = ToEigen(stiffnessMx.KJ, dofs);
  auto kKK = ToEigen(stiffnessMx.KK, dofs);
  auto cmatE = ToEigen(cmat, {pde.dofU, pde.dofV});

  stiffnessMxMod = kJJ;
  stiffnessMxMod -= cmatE.transpose() * kKJ;
  stiffnessMxMod -= kJK * cmatE;
  stiffnessMxMod += cmatE.transpose() * kKK * cmatE;

  // ***********************************
  //    Assemble load vector
  // ***********************************

  GlobalDofVector loadVector;

  loadVector += asmbl.BuildVector(neumannCellGroup, {pde.dofU},
                                  [&](const CellIpData &cipd) {
                                    return pde.NeumannLoadMechanical(
                                        cipd, Eigen::Vector3d::Zero());
                                  });

  loadVector += asmbl.BuildVector(neumannCellGroup, {pde.dofV},
                                  [&](const CellIpData &cipd) {
                                    return pde.NeumannLoadElectrical(cipd, 0.);
                                  });

  // Compute modified load vector

  // ***********************************
  //    Solve system
  // ***********************************

  // ***********************************
  //    Visualize
  // ***********************************

  NuTo::Visualize::Visualizer visualize(
      volumeCellGroup,
      NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryBrick(
          order + 1, Visualize::LOBATTO)));
  visualize.DofValues(pde.dofU);
  visualize.DofValues(pde.dofV);
  visualize.WriteVtuFile("Piezoelectricity.vtu");
}
