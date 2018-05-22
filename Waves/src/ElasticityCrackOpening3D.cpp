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

/* Rectangular Plate, linear isotropic homogeneous elasticity
 * Circular crack opening.
 */
int main(int argc, char *argv[]) {

  // *********************************
  //      Geometry parameter
  // *********************************

  MeshGmsh gmsh("plateH0.75_angle45_L0.4.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto backCrackFace = gmsh.GetPhysicalGroup("BackCrackFace");
  auto frontCrackFace = gmsh.GetPhysicalGroup("FrontCrackFace");
  auto crackBoundary = Unite(frontCrackFace, backCrackFace);

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

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 1.0e-6;
  double stepSize = 0.006e-6;
  int numSteps = 50000;

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

  int order = 2;

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
  auto domainIntType = CreateLobattoIntegrationType(
      domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage domainCells;
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

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  Eigen::VectorXd massMxMod = lumpedMassMx[dof1];

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  DofMatrixSparse<double> stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        return pde.Hessian0(cipd, 0.);
      });

  Eigen::SparseMatrix<double> stiffnessMxMod = stiffnessMx(dof1, dof1);

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

  NuTo::TimeIntegration::NY4NoVelocity<Eigen::VectorXd> ti;
  double t = 0.;

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

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {

    Eigen::VectorXd tmp =
        (-stiffnessMxMod * w + loadVectorMod * smearedStepFunction(t, tau));
    d2wdt2 = (tmp.array() / massMxMod.array()).matrix();
  };

  // ***********************
  // General infos
  // ***********************

  std::cout << "NumDofs: " << numDofs << std::endl;

  // ***********************
  // Solve
  // ***********************

  std::ofstream outfile;

  int plotcounter = 1;
  for (int i = 0; i < numSteps; i++) {
    t = i * stepSize;
    state = ti.DoStep(eq, state.first, state.second, t, stepSize);
    MergeResult(state.first);
    std::cout << i + 1 << std::endl;
    // output to data file
    outfile.open(resultDirectoryFull.string() + "topDisplacements.txt",
                 std::ios::app);
    int count = 0;
    for (double r : outRadius) {
      for (double phi : outAngles) {
        Eigen::VectorXd displ = myInterpolator.GetValue(count, dof1);
        outfile << displ[1] << "\t";
        count++;
      }
    }
    outfile << std::endl;
    outfile.close();
    // plot
    if ((i * 100) % numSteps == 0) {
      visualizeResult(resultDirectoryFull.string() +
                      "Crack3D_angle45_h0.75_NormalLoad2ndOrderSlow_" +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
