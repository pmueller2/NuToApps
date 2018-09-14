#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/integrands/DynamicMomentumBalance.h"

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

  double lX = 1.;
  double lY = 1.;
  double lZ = 1.;

  double meshSize = 0.1;

  double crackRadius = 0.2;
  double crackOpening = 0.0001; // half of crack opening distance

  // *********************************
  //    Generate Gmsh geo file
  // *********************************

  std::string gmshFileName = "SphereSegmentHoleOctant";

  std::ofstream out(gmshFileName + ".geo");
  out << "lX = " << lX << ";\n";
  out << "lY = " << lY << ";\n";
  out << "lZ = " << lZ << ";\n";
  out << "h = " << meshSize << ";\n";
  out << "c = " << crackRadius << ";\n";
  out << "d = " << crackOpening << ";\n";

  const std::string geoFileGeometry = R"(
r = (d*d + c*c)/(2.*d);

Point(100) = {0,  0,   0, h};
Point(1) = {d-r,  0,   0, h};
Point(2) = {d, 0,   0, h};
Point(3) = {lX, 0,   0, h};
Point(4) = {lX, lY,  0, h};
Point(5) = {0,  lY,  0, h};
Point(6) = {0,  c,  0, h};
Point(7) = {lX, 0, -lZ, h};
Point(8) = {lX, lY, -lZ, h};
Point(9) = {0, lY,-lZ, h};
Point(10) = {0, 0,-lZ, h};
Point(11) = {0, 0,-c, h};

Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};

Line(6) = {7,8};
Line(7) = {8,9};
Line(8) = {9,10};
Line(9) = {10,7};

Line(10) = {3,7};
Line(11) = {4,8};
Line(12) = {5,9};
Line(13) = {11,10};

Circle(30) = {2,1,6};
Circle(31) = {6,100,11};
Circle(32) = {11,1,2};

// Front
Line Loop(14) = {2,3,4,5,-30};
Plane Surface(1) = {14};

// Right
Line Loop(15) = {10,6,-11,-3};
Plane Surface(2) = {15};

// Left
Line Loop(16) = {-5,12,8,-13,-31};
Plane Surface(3) = {16};

// Top
Line Loop(17) = {11,7,-12,-4};
Plane Surface(4) = {17};

// Bottom
Line Loop(18) = {13,9,-10,-2,-32};
Plane Surface(5) = {18};

// Back
Line Loop(19) = {-8,-7,-6,-9};
Plane Surface(6) = {19};

// Crack
Line Loop(20) = {30,31,32};
Surface(7) = {20} In Sphere {1};

Surface Loop(8) = {1,2,3,4,5,6,7};
Volume(1) = {8};
)";

  const std::string geoFilePhysicalEntities = R"(
Physical Surface("Right") = {2};
Physical Surface("Left") = {3};
Physical Surface("Top") = {4};
Physical Surface("Front") = {1};
Physical Surface("Bottom") = {5};
Physical Surface("Back") = {6};
Physical Surface("Crack") = {7};

Physical Volume("Domain") = {1};
Mesh.SubdivisionAlgorithm = 2;
General.ExpertMode = 1;
)";

  out << geoFileGeometry;
  out << geoFilePhysicalEntities;
  out.close();

  // *********************************
  //    Generate Mesh and read it
  // *********************************

  std::cout << "Meshing..." << std::endl;
  system(("gmsh -3 -order 1 " + gmshFileName + ".geo -o " + gmshFileName +
          ".msh -v 2")
             .c_str());

  std::cout << "Read Mesh" << std::endl;
  MeshGmsh gmsh(gmshFileName + ".msh");
  MeshFem &mesh = gmsh.GetMeshFEM();
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto front = gmsh.GetPhysicalGroup("Front");
  auto back = gmsh.GetPhysicalGroup("Back");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto crack = gmsh.GetPhysicalGroup("Crack");
  auto boundary = Unite(left, bottom, right, top, front, back, crack);

  // **************************************
  // Result directory, filesystem
  // **************************************

  std::string resultDirectory = "/Elasticity3D_SphericalCrack/";
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

  // ***********************************
  //    Dofs, Interpolation
  // ***********************************

  double tau = 0.138;
  double stepSize = 0.004 / 2;
  int numSteps = 500 * 2;

  DofType dof1("Displacements", 3);

  double E = 1.;
  double nu = 0.3;
  double rho = 1.;

  Laws::LinearElastic<3> steel(E, nu);
  Integrands::DynamicMomentumBalance<3> pde(dof1, steel, rho);

  std::cout << "Create Interpolations" << std::endl;

  int order = 2;
  auto &ipol3D = mesh.CreateInterpolation(InterpolationBrickLobatto(order));
  AddDofInterpolation(&mesh, dof1, domain, ipol3D);

  auto &ipol2D = mesh.CreateInterpolation(InterpolationQuadLobatto(order));
  AddDofInterpolation(&mesh, dof1, boundary, ipol2D);

  mesh.AllocateDofInstances(dof1, 2);

  // ***********************************
  //    Set up integration, add cells
  // ***********************************

  int integrationOrder = order + 1;

  std::cout << "Create Cells" << std::endl;

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

  Group<NodeSimple> bottomNodes = GetNodes(bottom, dof1);
  Group<NodeSimple> leftNodes = GetNodes(left, dof1);
  Group<NodeSimple> frontNodes = GetNodes(front, dof1);

  // Symmetry conditions
  Constraint::Constraints constraints;
  constraints.Add(dof1, Constraint::Component(bottomNodes, {eDirection::Y}));
  constraints.Add(dof1, Constraint::Component(leftNodes, {eDirection::X}));
  constraints.Add(dof1, Constraint::Component(frontNodes, {eDirection::Z}));

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

  std::cout << "Assemble Mass" << std::endl;

  auto lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      domainCellGroup, {dof1},
      [&](const CellIpData &cipd) { return pde.Hessian2(cipd); });

  Eigen::SparseMatrix<double> massMxModFull =
      cmat.transpose() * lumpedMassMx[dof1].asDiagonal() * cmat;
  Eigen::VectorXd massMxMod = massMxModFull.diagonal();

  // ***********************************
  //    Assemble stiffness matrix
  // ***********************************

  std::cout << "Assemble Stiffness" << std::endl;

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
    double tractionMagnitude = -2. / tau * smearedHatFunction(tt, tau);
    Eigen::Vector3d normalTraction =
        cipd.GetJacobian().Normal() * tractionMagnitude;
    loadLocal[dof1] = N.transpose() * normalTraction;
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
      visualizeResult(resultDirectoryFull.string() + "Symmetric_" +
                      std::to_string(plotcounter));
      plotcounter++;
    }
  }
}
