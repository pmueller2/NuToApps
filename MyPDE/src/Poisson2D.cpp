#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTriangleLinear.h"

#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"

#include "nuto/visualize/AverageGeometries.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"

#include "../../NuToHelpers/ConstraintsHelper.h"
#include "../../NuToHelpers/MeshValuesTools.h"
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

bool CheckAllShapesEqual(NuTo::Group<ElementCollectionFem> g, DofType d) {
  eShape s = g.begin()->DofElement(d).GetShape().Enum();
  for (auto &elm : g) {
    if (s != elm.DofElement(d).GetShape().Enum())
      return false;
  }
  return true;
}

class PoissonEquation {
public:
  DofType dof1;
  DofType dof2;

  MeshFem &mesh;

  NuTo::Group<ElementCollectionFem> domain;
  NuTo::Group<ElementCollectionFem> dirichletBoundary;
  NuTo::Group<ElementCollectionFem> neumannBoundary;

  std::function<double(Eigen::Vector2d)> solution;
  std::function<Eigen::Vector2d(Eigen::Vector2d)> gradSolution;
  std::function<double(Eigen::Vector2d)> rightHandSide;

  PoissonEquation(std::function<double(Eigen::Vector2d)> u,
                  std::function<Eigen::Vector2d(Eigen::Vector2d)> du,
                  std::function<double(Eigen::Vector2d)> f, MeshFem &m)
      : dof1("Scalar", 1), dof2("Exact", 1), mesh(m), solution(u),
        gradSolution(du), rightHandSide(f) {

    // Add Interpolation
    AddDofInterpolation(&mesh, dof1);
    AddDofInterpolation(&mesh, dof2);
  }

  void SetSolutionDomain(NuTo::Group<ElementCollectionFem> g) { domain = g; }

  void SetDirichletBoundary(NuTo::Group<ElementCollectionFem> g) {
    dirichletBoundary = g;
  }

  void SetNeumannBoundary(NuTo::Group<ElementCollectionFem> g) {
    neumannBoundary = g;
  }

  void Solve(std::string filename) {
    // ***********************************
    //    Set up integration, add cells
    // ***********************************
    int integrationOrder = 3;

    // Domain cells
    auto domainIntType = CreateGaussIntegrationType(
        domain.begin()->DofElement(dof1).GetShape(), integrationOrder);
    if (!CheckAllShapesEqual(domain, dof1))
      throw Exception(
          __PRETTY_FUNCTION__,
          "Domain: Not all shapes equal - different integration types needed");
    CellStorage domainCells;
    auto domainCellGroup = domainCells.AddCells(domain, *domainIntType);

    // Boundary cells
    auto boundaryIntType = CreateGaussIntegrationType(
        neumannBoundary.begin()->DofElement(dof1).GetShape(), integrationOrder);
    if (!CheckAllShapesEqual(neumannBoundary, dof1))
      throw Exception(__PRETTY_FUNCTION__, "NeumannBoundary: Not all shapes "
                                           "equal - different integration "
                                           "types needed");
    CellStorage neumannCells;
    auto neumannCellGroup =
        neumannCells.AddCells(neumannBoundary, *boundaryIntType);

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

    // ***********************************
    //    Include Constraints
    // ***********************************

    Eigen::SparseMatrix<double> cmat =
        constraints.BuildUnitConstraintMatrix(dof1, numDofs);

    Eigen::SparseMatrix<double> stiffnessMxMod =
        cmat.transpose() * stiffnessMx(dof1, dof1) * cmat;

    // Compute modified load vector
    auto B = constraints.GetSparseGlobalRhs(dof1, numDofs, 0.);
    Eigen::VectorXd loadVectorMod =
        cmat.transpose() * (loadVector[dof1] - stiffnessMx(dof1, dof1) * B);

    // ***********************************
    //    Solve
    // ***********************************

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

void RectanglePoisson() {
  auto u = [](Eigen::Vector2d r) {
    return sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
  };
  auto du = [](Eigen::Vector2d r) {
    Eigen::Vector2d result;
    result[0] = 2 * M_PI * cos(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    result[1] = 2 * M_PI * sin(2 * M_PI * r[0]) * cos(2 * M_PI * r[1]);
    return result;
  };
  auto f = [](Eigen::Vector2d r) {
    double result =
        8 * M_PI * M_PI * sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    return result;
  };

  MeshGmsh gmsh("rectangle100x100.msh");
  auto top = gmsh.GetPhysicalGroup("Top");
  auto bottom = gmsh.GetPhysicalGroup("Bottom");
  auto left = gmsh.GetPhysicalGroup("Left");
  auto right = gmsh.GetPhysicalGroup("Right");
  auto domain = gmsh.GetPhysicalGroup("Domain");

  PoissonEquation eq(u, du, f, gmsh.GetMeshFEM());
  eq.SetSolutionDomain(domain);
  eq.SetDirichletBoundary(Unite(top, left));
  eq.SetNeumannBoundary(Unite(bottom, right));

  eq.Solve("Poisson2DRectangle.vtu");
}

void AnnulusPoisson() {
  auto u = [](Eigen::Vector2d r) {
    return sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
  };
  auto du = [](Eigen::Vector2d r) {
    Eigen::Vector2d result;
    result[0] = 2 * M_PI * cos(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    result[1] = 2 * M_PI * sin(2 * M_PI * r[0]) * cos(2 * M_PI * r[1]);
    return result;
  };
  auto f = [](Eigen::Vector2d r) {
    double result =
        8 * M_PI * M_PI * sin(2 * M_PI * r[0]) * sin(2 * M_PI * r[1]);
    return result;
  };

  MeshGmsh gmsh("annulus2ndOrder.msh");
  auto domain = gmsh.GetPhysicalGroup("Domain");
  auto inner = gmsh.GetPhysicalGroup("InnerBoundary");
  auto outer = gmsh.GetPhysicalGroup("OuterBoundary");

  PoissonEquation eq(u, du, f, gmsh.GetMeshFEM());
  eq.SetSolutionDomain(domain);
  eq.SetDirichletBoundary(inner);
  eq.SetNeumannBoundary(outer);

  eq.Solve("Poisson2DAnnulus.vtu");
}

void TestCheckAllShapesEqual() {
  MeshFem mesh;

  auto &nd1 = mesh.Nodes.Add(Eigen::Vector2d(0., 0.));
  auto &nd2 = mesh.Nodes.Add(Eigen::Vector2d(1., 0.));
  auto &nd3 = mesh.Nodes.Add(Eigen::Vector2d(1., 1.));
  auto &nd4 = mesh.Nodes.Add(Eigen::Vector2d(0., 1.));
  auto &nd5 = mesh.Nodes.Add(Eigen::Vector2d(0.5, 2.));

  auto &interpolationQ = mesh.CreateInterpolation(InterpolationQuadLinear());
  auto &interpolationT =
      mesh.CreateInterpolation(InterpolationTriangleLinear());
  auto &e0 = mesh.Elements.Add({{{nd1, nd2, nd3, nd4}, interpolationQ}});
  auto &e1 = mesh.Elements.Add({{{nd3, nd4, nd5}, interpolationT}});

  DofType d("test", 1);
  AddDofInterpolation(&mesh, d);

  if (CheckAllShapesEqual(mesh.ElementsTotal(), d))
    std::cout << "Mesh1: all Shapes equal (wrong)." << std::endl;
  else
    std::cout << "Mesh1: Not all Shapes equal! (OK)" << std::endl;

  MeshFem mesh2 = UnitMeshFem::CreateQuads(5, 5);
  AddDofInterpolation(&mesh2, d);

  if (CheckAllShapesEqual(mesh2.ElementsTotal(), d))
    std::cout << "Mesh2: all Shapes equal (OK)." << std::endl;
  else
    std::cout << "Mesh2: Not all Shapes equal! (wrong)" << std::endl;
}

int main(int argc, char *argv[]) {
  // RectanglePoisson();
  AnnulusPoisson();
  // TestCheckAllShapesEqual();
}
