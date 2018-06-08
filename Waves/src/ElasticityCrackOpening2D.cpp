#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/base/Timer.h"

#include "../../MyTimeIntegration/NY4NoVelocity.h"
#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/MeshValuesTools.h"

#include "boost/filesystem.hpp"

#include <iostream>

using namespace NuTo;

double smearedStepFunction(double t, double tau) {
  double ot = M_PI * t / tau;
  if (ot > M_PI)
    return 1.;
  if (ot < 0.)
    return 0.;
  return 0.5 * (1. - cos(ot));
}

/* Rectangular domain, linear isotropic homogeneous elasticity
 * Crack opening
 */
int main(int argc, char *argv[]) {

  // *********************************
  //      Geometry parameter
  // *********************************

  MeshGmsh gmsh("Crack2D_2.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto crack = gmsh.GetPhysicalGroup("Crack");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto boundary = Unite(Unite(left, bottom), crack);

  // **************************************
  // Result directory, filesystem
  // **************************************

  std::string resultDirectory = "/ElasticWaves2D/";
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
  // OutputNodes/OutputData
  // **************************************

  // Set up a list of output coordinates
  double outRadius = 0.5;
  int numAngles = 10;

  Eigen::MatrixXd outputCoords(numAngles, 2);

  for (int i = 0; i < numAngles; i++) {
    double phi = i * M_PI / 2 / (numAngles - 1);
    outputCoords(i, 0) = outRadius * cos(phi);
    outputCoords(i, 1) = outRadius * sin(phi);
  }

  Tools::Interpolator myInterpolator(outputCoords, domain);

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 0.2e-6;
  double stepSize = 0.006e-6;
  int numSteps = 100;

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  Eigen::Vector2d crackLoad(1., 0.);

  DofType dof1("Displacements", 2);

  Laws::LinearElastic<2> steel(E, nu); // 2D default: plane stress
  Integrands::DynamicMomentumBalance<2> pde(dof1, steel, rho);

  int order = 3;
  auto &ipol2D = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol2D);

  auto &ipol1D = mesh.CreateInterpolation(InterpolationTrussLobatto(order));
  AddDofInterpolation(&mesh, dof1, boundary, ipol1D);

  mesh.AllocateDofInstances(dof1, 2);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  int integrationOrder = order + 1;

  // Domain cells
  auto domainIntType = CreateLobattoIntegrationType(
      domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage domainCells;
  auto domainCellGroup = domainCells.AddCells(domain, *domainIntType);

  // Boundary cells
  auto boundaryIntType = CreateLobattoIntegrationType(
      crack.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage crackCells;
  auto crackCellGroup = crackCells.AddCells(crack, *boundaryIntType);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  Group<NodeSimple> bottomNodes = GetNodes(bottom, dof1);
  Group<NodeSimple> leftNodes = GetNodes(left, dof1);

  Constraint::Constraints constraints;
  constraints.Add(dof1, Constraint::Component(bottomNodes, {eDirection::Y}));
  constraints.Add(dof1, Constraint::Component(leftNodes, {eDirection::X}));

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  int numDofs =
      dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

  Eigen::SparseMatrix<double> cmat =
      constraints.BuildUnitConstraintMatrix(dof1, numDofs);

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  Eigen::SparseMatrix<double> massMxModFull =
      cmat.transpose() * lumpedMassMx[dof1].asDiagonal() * cmat;
  Eigen::VectorXd massMxMod = massMxModFull.diagonal();

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  DofMatrixSparse<double> stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.Hessian0(cipd, 0.);
      });

  Eigen::SparseMatrix<double> stiffnessMxMod =
      cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    //    NuTo::Visualize::Visualizer visualize(domainCellGroup,
    //                                          NuTo::Visualize::AverageHandler());
    NuTo::Visualize::Visualizer visualize(
        domainCellGroup,
        NuTo::Visualize::VoronoiHandler(Visualize::VoronoiGeometryQuad(
            integrationOrder, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
    visualize.CellData(
        [&](const CellIpData cipd) {
          EngineeringStress<2> stress =
              steel.Stress(cipd.Apply(dof1, Nabla::Strain()), 0., cipd.Ids());
          return stress;
        },
        "stress");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  // ***********************************
  //    Solve
  // ***********************************

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

  Eigen::VectorXd femResult(numDofs);

  // Set initial data
  Eigen::VectorXd w0(dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd v0(dofInfo.numIndependentDofs[dof1]);

  w0.setZero();
  v0.setZero();

  auto MergeResult = [&mesh, &dof1](Eigen::VectorXd femResult) {
    for (auto &node : mesh.NodesTotal(dof1)) {
      for (int component = 0; component < dof1.GetNum(); component++) {
        int dofNr = node.GetDofNumber(component);
        node.SetValue(component, femResult[dofNr]);
      }
    };
  };

  auto state = std::make_pair(w0, v0);

  auto crackLoadFunc = [&](const CellIpData &cipd, double tt) {
    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;

    Eigen::Vector2d normalTraction = crackLoad * smearedStepFunction(tt, tau);

    loadLocal[dof1] = N.transpose() * normalTraction;
    return loadLocal;
  };

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
    // Compute load
    DofVector<double> boundaryLoad =
        asmbl.BuildVector(crackCellGroup, {dof1}, [&](const CellIpData cipd) {
          return crackLoadFunc(cipd, t);
        });
    // Include constraints
    auto B = constraints.GetSparseGlobalRhs(dof1, numDofs, t);
    Eigen::VectorXd loadVectorMod =
        cmat.transpose() * (boundaryLoad[dof1] - stiffnessMx(dof1, dof1) * B);

    d2wdt2 = (-stiffnessMxMod * w + loadVectorMod);
    d2wdt2 = d2wdt2.cwiseQuotient(massMxMod);
  };

  std::cout << "NumDofs: " << numDofs << std::endl;
  Timer timer("Solve");

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * stepSize;
    state = ti.DoStep(eq, state.first, state.second, t, stepSize);
    femResult =
        cmat * state.first +
        constraints.GetSparseGlobalRhs(dof1, numDofs, (i + 1) * stepSize);
    // output test
    //    for (int i = 0; i < outputCoords.rows(); i++) {
    //      std::cout << myInterpolator.GetValue(i, dof1) << std::endl;
    //    }
    std::cout << i + 1 << std::endl;
    //    if ((i * 100) % numSteps == 0) {
    //      MergeResult(femResult);
    //      visualizeResult(resultDirectoryFull.string() + "Crack2DNormalLoad_"
    //      +
    //                      std::to_string(plotcounter));
    //      plotcounter++;
    //    }
  }
}
