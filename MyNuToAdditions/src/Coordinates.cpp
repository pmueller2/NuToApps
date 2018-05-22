#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTriangleLinear.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include "../../MyPDE/src/PolarCoordinates.h"
#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/MeshValuesTools.h"
#include "../../NuToHelpers/MyDifferentialOperators.h"
#include "../../NuToHelpers/PoissonTypeProblem.h"

#include <iostream>

using namespace NuTo;

/* Solves the Poisson equation
 *     -Delta u = f
 *
 * in 2D, given: solution u, its gradient du and the right hand side f
 *
 *
*/

Eigen::Vector2d polarToCartesian(Eigen::Vector2d vals) {
  double r = vals[0];
  double phi = vals[1];
  double x = r * cos(phi);
  double y = r * sin(phi);
  return Eigen::Vector2d(x, y);
}

Eigen::Vector2d cartesianToPolar(Eigen::Vector2d vals) {
  double x = vals[0];
  double y = vals[1];
  double r = sqrt(x * x + y * y);
  double phi = atan2(y, x);
  return Eigen::Vector2d(r, phi);
}

class PoissonEquation {
public:
  DofType dof1;
  DofType dof2;

  MeshFem &mesh;

  CellStorage domainCells;
  NuTo::Group<CellInterface> domainCellGroup;

  CellStorage neumannCells;
  NuTo::Group<CellInterface> neumannCellGroup;

  NuTo::Group<ElementCollectionFem> domain;
  NuTo::Group<ElementCollectionFem> dirichletBoundary;
  NuTo::Group<ElementCollectionFem> neumannBoundary;

  std::function<double(Eigen::Vector2d)> solution = [](Eigen::Vector2d r) {
    return sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
  };

  std::function<Eigen::Vector2d(Eigen::Vector2d)> gradSolution =
      [](Eigen::Vector2d r) {
        Eigen::Vector2d result;
        result[0] = 2 * M_PI * cos(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
        result[1] = 2 * M_PI * sin(2 * M_PI * r[0]) * cos(2 * M_PI * r[1]);
        return result;
      };

  std::function<double(Eigen::Vector2d)> rightHandSide = [](Eigen::Vector2d r) {
    double result =
        8 * M_PI * M_PI * sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    return result;
  };

  PoissonEquation(MeshFem &m) : dof1("Scalar", 1), dof2("Exact", 1), mesh(m) {

    // Add Interpolation
    AddDofInterpolation(&mesh, dof1);
    AddDofInterpolation(&mesh, dof2);
    domain = mesh.ElementsTotal();
  }

  void SetSolutionDomain(NuTo::Group<ElementCollectionFem> g) { domain = g; }

  void SetDirichletBoundary(NuTo::Group<ElementCollectionFem> g) {
    dirichletBoundary = g;
  }

  void SetNeumannBoundary(NuTo::Group<ElementCollectionFem> g) {
    neumannBoundary = g;
  }

  void Solve() {
    int integrationOrder = 3;

    // Domain cells
    auto domainIntType = CreateGaussIntegrationType(
        domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
    domainCellGroup = domainCells.AddCells(domain, *domainIntType);

    // Boundary cells
    auto boundaryIntType = CreateGaussIntegrationType(
        neumannBoundary.begin()->DofElement(dof1).GetShape(), integrationOrder);
    neumannCellGroup = neumannCells.AddCells(neumannBoundary, *boundaryIntType);

    // ******************************
    //      Set Dirichlet boundary
    // ******************************

    Constraint::Constraints constraints;
    if (!dirichletBoundary.Empty()) {
      constraints.Add(dof1, Constraint::SetDirichletBoundaryNodes(
                                dof1, dirichletBoundary, solution));
    }

    // ***********************************
    //    Assembly
    // ***********************************

    NuTo::Integrands::PoissonTypeProblem<2> pde(dof1);

    DofInfo dofInfo =
        DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);
    int numDofs =
        dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

    SimpleAssembler asmbl = SimpleAssembler(dofInfo);

    DofMatrixSparse<double> stiffnessMx =
        asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.StiffnessMatrix(cipd);
        });

    DofVector<double> loadVector =
        asmbl.BuildVector(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.LoadVector(cipd, rightHandSide);
        });

    DofVector<double> boundaryloadVector = asmbl.BuildVector(
        neumannCellGroup, {dof1}, [&](const CellIpData &cipd) {
          return pde.NeumannLoadWithGivenGradient(cipd, gradSolution);
        });

    loadVector += boundaryloadVector;

    Eigen::SparseMatrix<double> cmat =
        constraints.BuildUnitConstraintMatrix(dof1, numDofs);

    Eigen::SparseMatrix<double> stiffnessMxMod =
        cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

    // Compute modified load vector
    auto B = constraints.GetSparseGlobalRhs(dof1, numDofs, 0.);
    Eigen::VectorXd loadVectorMod =
        cmat.transpose() * (loadVector[dof1] - stiffnessMx(dof1, dof1) * B);

    // Compute Independent Dofs
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(stiffnessMxMod);
    if (solver.info() != Eigen::Success) {
      throw Exception("decomposition failed");
    }
    Eigen::VectorXd result = solver.solve(loadVectorMod);
    if (solver.info() != Eigen::Success) {
      throw Exception("solve failed");
    }
    // Additionally compute dependent Dofs
    Eigen::VectorXd femResult(numDofs);
    femResult = cmat * result + B;

    // Merge
    for (auto &node : mesh.NodesTotal(dof1)) {
      node.SetValue(0, femResult[node.GetDofNumber(0)]);
    }
  }

  void Plot(std::string filename) {
    // ***********************************
    //    Visualize
    // ***********************************

    NuTo::Visualize::Visualizer visualize(domainCellGroup,
                                          NuTo::Visualize::AverageHandler());
    visualize.DofValues(dof1);

    Tools::SetValues(domain, dof2, solution);

    visualize.DofValues(dof2);
    visualize.WriteVtuFile(filename);
  }
};

class PoissonEquationPolar {
public:
  DofType dof1;
  DofType dof2;

  MeshFem &mesh;

  CellStorage domainCells;
  NuTo::Group<CellInterface> domainCellGroup;

  CellStorage neumannCells;
  NuTo::Group<CellInterface> neumannCellGroup;

  NuTo::Group<ElementCollectionFem> domain;
  NuTo::Group<ElementCollectionFem> dirichletBoundary;
  NuTo::Group<ElementCollectionFem> neumannBoundary;

  std::function<double(Eigen::Vector2d)> solution = [](Eigen::Vector2d u) {
    auto r = polarToCartesian(u);
    return sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
  };

  std::function<Eigen::Vector2d(Eigen::Vector2d)> gradSolution =
      [](Eigen::Vector2d u) {
        auto r = polarToCartesian(u);
        Eigen::Vector2d result;
        result[0] = 2 * M_PI * cos(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
        result[1] = 2 * M_PI * sin(2 * M_PI * r[0]) * cos(2 * M_PI * r[1]);
        return result;
      };

  std::function<double(Eigen::Vector2d)> rightHandSide = [](Eigen::Vector2d u) {
    auto r = polarToCartesian(u);
    double result =
        8 * M_PI * M_PI * sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    return result;
  };

  PoissonEquationPolar(MeshFem &m)
      : dof1("Scalar", 1), dof2("Exact", 1), mesh(m) {

    // Add Interpolation
    AddDofInterpolation(&mesh, dof1);
    AddDofInterpolation(&mesh, dof2);
    domain = mesh.ElementsTotal();
  }

  void SetSolutionDomain(NuTo::Group<ElementCollectionFem> g) { domain = g; }

  void SetDirichletBoundary(NuTo::Group<ElementCollectionFem> g) {
    dirichletBoundary = g;
  }

  void SetNeumannBoundary(NuTo::Group<ElementCollectionFem> g) {
    neumannBoundary = g;
  }

  void Solve() {
    int integrationOrder = 3;

    // Domain cells
    auto domainIntType = CreateGaussIntegrationType(
        domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
    domainCellGroup = domainCells.AddCells(domain, *domainIntType);

    // Boundary cells
    std::unique_ptr<IntegrationTypeBase> boundaryIntType;
    if (!neumannBoundary.Empty())
      boundaryIntType = CreateGaussIntegrationType(
          neumannBoundary.begin()->DofElement(dof1).GetShape(),
          integrationOrder);
    neumannCellGroup = neumannCells.AddCells(neumannBoundary, *boundaryIntType);

    // ******************************
    //      Set Dirichlet boundary
    // ******************************

    Constraint::Constraints constraints;
    if (!dirichletBoundary.Empty()) {
      constraints.Add(dof1, Constraint::SetDirichletBoundaryNodes(
                                dof1, dirichletBoundary, solution));
    }

    // ***********************************
    //    Assembly
    // ***********************************

    DofInfo dofInfo =
        DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);
    int numDofs =
        dofInfo.numIndependentDofs[dof1] + dofInfo.numDependentDofs[dof1];

    SimpleAssembler asmbl = SimpleAssembler(dofInfo);
    PolarCoordinates polarCoords;

    DofMatrixSparse<double> stiffnessMx =
        asmbl.BuildMatrix(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          // Eigen::MatrixXd B = cipd.B(dof1, Nabla::Gradient());
          CoordinateSystem<2> localBasisSystem =
              polarCoords.GetNaturalCOOS(cipd.GlobalCoordinates());
          Eigen::MatrixXd B =
              cipd.B(dof1, Nabla::GradientCOOS2D(localBasisSystem));
          DofMatrix<double> stiffnessLocal;
          double polarJacobiDeterminant = localBasisSystem.GetDetJ();
          stiffnessLocal(dof1, dof1) = B.transpose() * B;
          return stiffnessLocal * polarJacobiDeterminant;
        });

    DofVector<double> loadVector =
        asmbl.BuildVector(domainCellGroup, {dof1}, [&](const CellIpData &cipd) {
          Eigen::MatrixXd N = cipd.N(dof1);
          DofVector<double> loadLocal;
          CoordinateSystem<2> localBasisSystem =
              polarCoords.GetNaturalCOOS(cipd.GlobalCoordinates());
          loadLocal[dof1] =
              N.transpose() * rightHandSide(cipd.GlobalCoordinates());
          double polarJacobiDeterminant = localBasisSystem.GetDetJ();
          return loadLocal * polarJacobiDeterminant;
        });

    //    DofVector<double> boundaryloadVector = asmbl.BuildVector(
    //        neumannCellGroup, {dof1}, [&](const CellIpData &cipd) {
    //          Eigen::MatrixXd N = cipd.N(dof1);
    //          DofVector<double> loadLocal;

    //          double normalComponent = gradSolution(cipd.GlobalCoordinates())
    //                                       .dot(cipd.GetJacobian().Normal());

    //          loadLocal[dof1] = N.transpose() * normalComponent;
    //          return loadLocal;
    //        });

    //    loadVector += boundaryloadVector;

    Eigen::SparseMatrix<double> cmat =
        constraints.BuildUnitConstraintMatrix(dof1, numDofs);

    Eigen::SparseMatrix<double> stiffnessMxMod =
        cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

    // Compute modified load vector
    auto B = constraints.GetSparseGlobalRhs(dof1, numDofs, 0.);
    Eigen::VectorXd loadVectorMod =
        cmat.transpose() * (loadVector[dof1] - stiffnessMx(dof1, dof1) * B);

    // Compute Independent Dofs
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(stiffnessMxMod);
    if (solver.info() != Eigen::Success) {
      throw Exception("decomposition failed");
    }
    Eigen::VectorXd result = solver.solve(loadVectorMod);
    if (solver.info() != Eigen::Success) {
      throw Exception("solve failed");
    }
    // Additionally compute dependent Dofs
    Eigen::VectorXd femResult(numDofs);
    femResult = cmat * result + B;

    // Merge
    for (auto &node : mesh.NodesTotal(dof1)) {
      node.SetValue(0, femResult[node.GetDofNumber(0)]);
    }
  }

  void SetExact() { Tools::SetValues(domain, dof2, solution); }

  void Plot(std::string filename) {

    NuTo::Visualize::Visualizer visualize(domainCellGroup,
                                          NuTo::Visualize::AverageHandler());
    visualize.DofValues(dof1);

    visualize.DofValues(dof2);
    visualize.WriteVtuFile(filename);
  }
};

// **********************************************
//  Solves Poisson equation with known solution
// **********************************************
void RectanglePoisson() {

  MeshGmsh gmsh("rectangle100x100.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();

  for (auto &node : mesh.NodesTotal()) {
    double rMin = 0.01;
    double rMax = 1.0;
    double r = rMin + (rMax - rMin) * node.GetValues()[0];
    double phi = 2 * M_PI * node.GetValues()[1];
    node.SetValues(polarToCartesian(Eigen::Vector2d(r, phi)));
  }

  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  PoissonEquation eq(mesh);
  eq.SetSolutionDomain(domain);
  eq.SetDirichletBoundary(Unite(top, left));
  eq.SetNeumannBoundary(Unite(bottom, right));

  eq.Solve();
  eq.Plot("Poisson2DRectangle.vtu");
}

// *******************************************************************
//  Solves Poisson equation with known solution in polar coordinates
// *******************************************************************
void PolarPoisson() {

  MeshGmsh gmsh("rectangle100x100.msh");
  MeshFem &mesh = gmsh.GetMeshFEM();

  for (auto &node : mesh.NodesTotal()) {
    Eigen::Vector2d vals = node.GetValues();
    double rMin = 0.01;
    double rMax = 1.0;
    double r = rMin + (rMax - rMin) * vals[0];
    double phi = 2 * M_PI * vals[1];
    node.SetValues(Eigen::Vector2d(r, phi));
  }

  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  PoissonEquationPolar eq(mesh);
  eq.SetSolutionDomain(domain);
  // eq.SetDirichletBoundary(Unite(top, left));
  // eq.SetNeumannBoundary(Unite(bottom, right));

  eq.SetDirichletBoundary(Unite(Unite(top, left), Unite(bottom, right)));
  eq.SetNeumannBoundary(Intersection(left, right));

  eq.Solve();

  eq.SetExact();

  for (auto &node : mesh.NodesTotal()) {
    node.SetValues(polarToCartesian(node.GetValues()));
  }

  eq.Plot("Poisson2DPolar.vtu");
}

int main(int argc, char *argv[]) {
  // RectanglePoisson();
  PolarPoisson();
}
