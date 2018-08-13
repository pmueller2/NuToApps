#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
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

/* Rectangular Plate, linear isotropic homogeneous elasticity
 * Circular crack opening.
 */
int main(int argc, char *argv[]) {
  std::cout << "large mesh" << std::endl;

  // *********************************
  //      Geometry parameter
  // *********************************

  Timer timer("Load Mesh");

  MeshGmsh gmsh("plateH0.1_angle45_L0.4small.msh");

  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto backCrackFace = gmsh.GetPhysicalGroup("BackCrackFace");
  auto frontCrackFace = gmsh.GetPhysicalGroup("FrontCrackFace");
  auto crackBoundary = Unite(frontCrackFace, backCrackFace);

  timer.Reset("Output directory stuff");
  std::cout << std::flush;

  // **************************************
  // Result directory, filesystem
  // **************************************

  std::string resultDirectory = "/ElasticWaves3D/";
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

  timer.Reset("My Interpolator");
  std::cout << std::flush;

  double centerX = 5.;
  double centerZ = 5.;

  // Set up a list of output coordinates

  int numOutRadius = 5;
  double minR = 1.;
  double maxR = 4.5;

  std::vector<double> outRadius;
  for (int i = 0; i < numOutRadius; i++) {
    outRadius.push_back(minR + i * (maxR - minR) / (numOutRadius - 1));
  }

  // Angles in degree
  int numOutAngles = 10;
  double minPhi = 0.0;
  double maxPhi = 90.0;

  std::vector<double> outAngles;
  for (int i = 0; i < numOutAngles; i++) {
    outAngles.push_back(2 * M_PI / 360. *
                        (minPhi + i * (maxPhi - minPhi) / (numOutAngles - 1)));
  }

  Eigen::MatrixXd outputCoords(numOutAngles * numOutRadius, 3);

  int count = 0;
  for (double r : outRadius) {
    for (double phi : outAngles) {
      outputCoords(count, 0) = r * cos(phi) + centerX;
      outputCoords(count, 1) = 1.;
      outputCoords(count, 2) = r * sin(phi) + centerZ;
      count++;
    }
  }

  Tools::Interpolator myInterpolator(outputCoords, top);

  timer.Reset("Interpolation");
  std::cout << std::flush;

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 1.0e-6;
  double stepSize = 0.006e-6;
  int numSteps = 10000;
  int plotSteps = 100;

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  double crackAngle = 0.25 * M_PI;
  double crackRadius = 0.2;

  double crackArea = M_PI * crackRadius * crackRadius;
  double loadMagnitude = 1. / crackArea;

  // Load on Front crack face
  Eigen::Vector3d crackLoad(0., -sin(crackAngle), cos(crackAngle));
  crackLoad *= loadMagnitude;

  DofType dof1("Displacements", 3);

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  int order = 1;

  auto &ipol3D = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol3D);

  auto &ipol2D = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, crackBoundary, ipol2D);
  AddDofInterpolation(&mesh, dof1, top, ipol2D);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  int integrationOrder = order + 1;

  // Domain cells
  timer.Reset("Create integration type");
  std::cout << std::flush;

  auto domainIntType = CreateLobattoIntegrationType(
      domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage domainCells;

  timer.Reset("Create cell group");
  std::cout << std::flush;

  auto domainCellGroup = domainCells.AddCells(domain, *domainIntType);

  // Boundary cells
  auto boundaryIntType = CreateLobattoIntegrationType(
      crackBoundary.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage crackCellsFront;
  auto crackCellGroupFront =
      crackCellsFront.AddCells(frontCrackFace, *boundaryIntType);

  CellStorage crackCellsBack;
  auto crackCellGroupBack =
      crackCellsBack.AddCells(backCrackFace, *boundaryIntType);

  // *********************************************
  //      No Dirichlet boundary, Numbering
  // *********************************************

  Constraint::Constraints constraints;

  DofInfo dofInfo =
      DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

  // ************************************
  //   Assemble constant mass matrix
  // ************************************

  int numDofs =
      dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

  // ***********************
  // General infos
  // ***********************

  std::cout << "NumDofs: " << numDofs << std::endl;

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  Eigen::VectorXd massMxMod = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); })[dof1];

  timer.Reset("Mass assembly");
  std::cout << std::flush;

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  Eigen::SparseMatrix<double, Eigen::RowMajor> stiffnessMxMod =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.Hessian0(cipd, 0.);
      })(dof1, dof1);

  timer.Reset("Stiffness assembly");
  std::cout << std::flush;

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        domainCellGroup,
        NuTo::Visualize::VoronoiHandler(Visualize::VoronoiGeometryBrick(
            integrationOrder, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
    visualize.CellData(
        [&](const CellIpData cipd) {
          EngineeringStress<3> stress =
              steel.Stress(cipd.Apply(dof1, Nabla::Strain()), 0., cipd.Ids());
          return stress;
        },
        "stress");
    visualize.WriteVtuFile(filename + ".vtu");
  };

  // ***********************************
  //    Solve
  // ***********************************

  double t = 0.;

  // Set initial data
  Eigen::VectorXd w0(dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd v0(dofInfo.numIndependentDofs[dof1]);

  w0.setZero();
  v0.setZero();

  auto MergeResult = [&mesh, &dof1](Eigen::VectorXd &femResult) {
    for (auto &node : mesh.NodesTotal(dof1)) {
      for (int component = 0; component < dof1.GetNum(); component++) {
        int dofNr = node.GetDofNumber(component);
        node.SetValue(component, femResult[dofNr]);
      }
    };
  };

  auto crackLoadFunc = [&](const CellIpData &cipd, double tt) {
    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;

    Eigen::Vector3d normalTraction = crackLoad * smearedStepFunction(tt, tau);

    loadLocal[dof1] = N.transpose() * normalTraction;
    return loadLocal;
  };

  // Compute load
  DofVector<double> boundaryLoadFront = asmbl.BuildVector(
      crackCellGroupFront, {dof1},
      [&](const CellIpData cipd) { return crackLoadFunc(cipd, tau * 2); });
  DofVector<double> boundaryLoadBack =
      asmbl.BuildVector(crackCellGroupBack, {dof1}, [&](const CellIpData cipd) {
        return crackLoadFunc(cipd, tau * 2);
      });
  Eigen::VectorXd loadVectorMod =
      boundaryLoadFront[dof1] - boundaryLoadBack[dof1];

  // ***********************
  // Solve
  // ***********************

  timer.Reset("Solve");
  std::cout << std::flush;

  const Eigen::Matrix4d a = (Eigen::Matrix4d() << 0., 0., 0., 0., //
                             0.5, 0., 0., 0.,                     //
                             0.5, 0.5, 0., 0.,                    //
                             0.0, 0.0, 1., 0.                     //
                             ).finished();

  const Eigen::Vector4d b =
      (Eigen::Vector4d() << 1. / 6., 1. / 3., 1. / 3., 1. / 6.).finished();

  const Eigen::Vector4d c = (Eigen::Vector4d() << 0., 0.5, 0.5, 1.).finished();

  Eigen::VectorXd wOld(2 * dofInfo.numIndependentDofs[dof1]);
  wOld << w0, v0;
  Eigen::VectorXd wNew(2 * dofInfo.numIndependentDofs[dof1]);
  Eigen::VectorXd wni(2 * dofInfo.numIndependentDofs[dof1]);
  std::vector<Eigen::VectorXd> k(c.size());
  k[0].resize(2 * dofInfo.numIndependentDofs[dof1]);
  k[1].resize(2 * dofInfo.numIndependentDofs[dof1]);
  k[2].resize(2 * dofInfo.numIndependentDofs[dof1]);
  k[3].resize(2 * dofInfo.numIndependentDofs[dof1]);

  Eigen::VectorXd femResult(dofInfo.numIndependentDofs[dof1]);

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * stepSize;
    wNew = wOld;
    for (std::size_t i = 0; i < c.size(); i++) {
      double tRK = t + c[i] * stepSize;
      wni = wOld;
      for (std::size_t j = 0; j < i; j++) {
        if (a(i, j) != 0)
          wni += stepSize * a(i, j) * k[j];
      }
      // Here the RHS is evaluated
      Timer tm("RHS");
      k[i].head(dofInfo.numIndependentDofs[dof1]) =
          wni.tail(dofInfo.numIndependentDofs[dof1]);
      k[i].tail(dofInfo.numIndependentDofs[dof1]) =
          -stiffnessMxMod * wni.head(dofInfo.numIndependentDofs[dof1]) +
          loadVectorMod * smearedStepFunction(t, tau);
      k[i].tail(dofInfo.numIndependentDofs[dof1]).cwiseQuotient(massMxMod);
      wNew += stepSize * b[i] * k[i];
    }
    wOld = wNew;

    std::cout << i + 1 << std::endl;

    if (i % plotSteps == 0) {
      femResult = wNew.head(dofInfo.numIndependentDofs[dof1]);
      MergeResult(femResult);
      visualizeResult(resultDirectoryFull.string() +
                      "Crack3D_angle45_h0.1_smallNormalLoad2ndOrderSlow_" +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
