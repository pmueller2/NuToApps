#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/cell/CellIpData.h"
#include "nuto/mechanics/cell/DifferentialOperators.h"
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
#include "../../NuToHelpers/NiceLookingFunctions.h"

#include "boost/filesystem.hpp"

#include <fstream>
#include <iostream>

using namespace NuTo;

/* Rectangular domain, scalar wave equation
 */
int main(int argc, char *argv[]) {

  // *********************************
  //       Set up geometry data
  // *********************************

  double width = 1.;
  double length = 1.;

  double holeX = 0.02; // ellipse main axis in X direction
  double holeY = 0.2;  // ellipse main axis in Y direction

  double meshSize = length / 50;

  // *********************************
  //    Generate Gmsh geo file
  // *********************************

  std::string gmshFileName = "EllipticalHoleQuarter";

  std::ofstream out(gmshFileName + ".geo");
  out << "lX = " << length << ";\n";
  out << "lY = " << width << ";\n";
  out << "h = " << meshSize << ";\n";
  out << "cX = " << holeX << ";\n";
  out << "cY = " << holeY << ";\n";

  const std::string geoFileGeometry = R"(
Point(1) = {0,  0,   0, h};
Point(2) = {cX, 0,   0, h};
Point(3) = {lX, 0,   0, h};
Point(4) = {lX, lY,  0, h};
Point(5) = {0,  lY,  0, h};
Point(6) = {0,  cY,  0, h};

Ellipse(1) = {6,1,2,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};

Line Loop(6) = {1,2,3,4,5};
Plane Surface(1) = {6};

Recombine Surface(1);
)";

  const std::string geoFilePhysicalEntities = R"(
Physical Line("Bottom") = {2};
Physical Line("Top") = {4};
Physical Line("Left") = {5};
Physical Line("Right") = {3};
Physical Line("Crack") = {1};
Physical Surface("Domain") = {1};
)";

  out << geoFileGeometry;
  out << geoFilePhysicalEntities;
  out.close();

  // *********************************
  //    Generate Mesh and read it
  // *********************************

  std::cout << "Meshing..." << std::endl;
  system(("gmsh -2 -order 1 " + gmshFileName + ".geo -o " + gmshFileName +
          ".msh -v 2")
             .c_str());

  MeshGmsh gmsh(gmshFileName + ".msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto top = gmsh.GetPhysicalGroup("Top");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto crack = gmsh.GetPhysicalGroup("Crack");
  auto boundary = Unite(left, bottom, right, top, crack);

  // **************************************
  // Result directory, filesystem
  // **************************************

  std::string resultDirectory = "/Waves2D_EllipticalCrack/";
  bool overwriteResultDirectory = false;

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

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 0.138;
  double stepSize = 0.004 / 2;
  int numSteps = 500 * 2;

  DofType dof1("Scalar", 1);

  int order = 2;
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

  // Crack boundary cells
  auto boundaryIntType = CreateLobattoIntegrationType(
      left.begin()->DofElement(dof1).GetShape(), integrationOrder);
  CellStorage crackBoundaryCells;
  auto crackBoundaryCellGroup =
      crackBoundaryCells.AddCells(crack, *boundaryIntType);

  // ******************************
  //      Set Dirichlet boundary
  // ******************************

  // None
  Constraint::Constraints constraints;

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
      domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        Eigen::MatrixXd N = cipd.N(dof1);
        DofMatrix<double> massLocal;
        massLocal(dof1, dof1) = N.transpose() * N;
        return massLocal;
      });

  Eigen::SparseMatrix<double> massMxModFull =
      cmat.transpose() * lumpedMassMx[dof1].asDiagonal() * cmat;
  Eigen::VectorXd massMxMod = massMxModFull.diagonal();

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  DofMatrixSparse<double> stiffnessMx =
      asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
        Eigen::MatrixXd B = cipd.B(dof1, Nabla::Gradient());
        DofMatrix<double> stiffnessLocal;
        stiffnessLocal(dof1, dof1) = B.transpose() * B;
        return stiffnessLocal;
      });

  Eigen::SparseMatrix<double> stiffnessMxMod =
      cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

  // *********************************
  //      Visualize
  // *********************************

  auto visualizeResult = [&](std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        domainCellGroup,
        NuTo::Visualize::VoronoiHandler(Visualize::VoronoiGeometryQuad(
            integrationOrder, Visualize::LOBATTO)));
    visualize.DofValues(dof1);
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

  auto loadFunc = [&](const CellIpData &cipd, double tt) {
    Eigen::MatrixXd N = cipd.N(dof1);
    DofVector<double> loadLocal;

    loadLocal[dof1] = N.transpose() * 2. / tau * smearedHatFunction(tt, tau);
    return loadLocal;
  };

  auto eq = [&](const Eigen::VectorXd &w, Eigen::VectorXd &d2wdt2, double t) {
    // Compute load
    DofVector<double> boundaryLoad = asmbl.BuildVector(
        crackBoundaryCellGroup, {dof1},
        [&](const CellIpData cipd) { return loadFunc(cipd, t); });
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

    std::cout << i + 1 << std::endl;
    if ((i * 100) % numSteps == 0) {
      MergeResult(femResult);
      visualizeResult(resultDirectoryFull.string() + "Ellipse_" +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
