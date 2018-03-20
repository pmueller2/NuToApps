#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/constraints/ConstraintCompanion.h"
#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/dofs/DofNumbering.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/visualize/UnstructuredGrid.h"
#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include <iostream>

#include "../../NuToHelpers/PoissonTypeProblem.h"
#include "/usr/include/eigen3/unsupported/Eigen/ArpackSupport"

using namespace NuTo;

void ExtractNodeVals(Eigen::VectorXd &femResult, Group<NodeSimple> nodes) {
  for (NodeSimple &node : nodes) {
    for (int i = 0; i < node.GetNumValues(); i++) {
      int dofNr = node.GetDofNumber(i);
      femResult[dofNr] = node.GetValues()(i);
    }
  }
}

void MergeNodeVals(const Eigen::VectorXd &femResult, Group<NodeSimple> nodes) {
  for (NodeSimple &node : nodes) {
    for (int i = 0; i < node.GetNumValues(); i++) {
      int dofNr = node.GetDofNumber(i);
      node.SetValue(i, femResult[dofNr]);
    }
  };
}

class TestProblem {
public:
  MeshFem mesh;
  const InterpolationSimple &interpolation2D;
  DofType dof1;
  Constraint::Constraints constraints;
  Eigen::SparseMatrix<double> cmat;
  SimpleAssembler asmbl;
  IntegrationTypeTensorProduct<2> integrationType2D;
  Eigen::SparseMatrix<double> stiffnessMxMod;
  Eigen::VectorXd massMxMod;
  Eigen::VectorXd femResult;
  Eigen::VectorXd femVelocities;
  boost::ptr_vector<CellInterface> cells;
  Group<CellInterface> cellGroup;
  int numDofsJ;
  int numDofsK;
  int numDofs;
  int mOrder;

  TestProblem(int numElms, int order)
      : mesh(std::move(UnitMeshFem::CreateQuads(numElms, numElms))),
        interpolation2D(
            mesh.CreateInterpolation(InterpolationQuadLobatto(order))),
        dof1("Scalar", 1),
        integrationType2D(order + 2, eIntegrationMethod::LOBATTO),
        mOrder(order) {

    AddDofInterpolation(&mesh, dof1, interpolation2D);

    constraints.Add(dof1, Constraint::Value(mesh.NodeAtCoordinate(
                              Eigen::Vector2d(0.5, 0.5), dof1, 1.e-7)));

    DofInfo dofInfo =
        DofNumbering::Build(mesh.NodesTotal(dof1), dof1, constraints);

    numDofsJ = dofInfo.numIndependentDofs[dof1];
    numDofsK = dofInfo.numDependentDofs[dof1];
    numDofs = numDofsJ + numDofsK;

    cmat = constraints.BuildConstraintMatrix(dof1, numDofsJ);
    asmbl = SimpleAssembler(dofInfo);

    int cellId = 0;
    for (ElementCollection &element : mesh.Elements) {
      cells.push_back(new Cell(element, integrationType2D, cellId++));
    }
    for (CellInterface &c : cells) {
      cellGroup.Add(c);
    }

    // Matrix assembly
    using namespace std::placeholders;
    Integrands::PoissonTypeProblem<2> equations(dof1);
    auto stiffnessF = std::bind(
        &Integrands::PoissonTypeProblem<2>::StiffnessMatrix, &equations, _1);
    auto massF = std::bind(&Integrands::PoissonTypeProblem<2>::MassMatrix,
                           &equations, _1);

    auto stiffnessMx = asmbl.BuildMatrix(cellGroup, {dof1}, stiffnessF);
    auto lumpedMassMx =
        asmbl.BuildDiagonallyLumpedMatrix(cellGroup, {dof1}, massF);

    // Modify because of constraints
    auto kJJ = stiffnessMx.JJ(dof1, dof1);
    auto kJK = stiffnessMx.JK(dof1, dof1);
    auto kKJ = stiffnessMx.KJ(dof1, dof1);
    auto kKK = stiffnessMx.KK(dof1, dof1);

    stiffnessMxMod = kJJ - cmat.transpose() * kKJ - kJK * cmat +
                     cmat.transpose() * kKK * cmat;

    auto mJ = lumpedMassMx.J[dof1];
    auto mK = lumpedMassMx.K[dof1];

    Eigen::SparseMatrix<double> massModifier =
        cmat.transpose() * mK.asDiagonal() * cmat;

    massMxMod = mJ + massModifier.diagonal();

    // Setup a solution vector
    femResult.resize(numDofs);
    femResult.setZero();
  }

  void Plot(std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        cellGroup,
        NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryQuad(
            5 * mOrder, Visualize::LOBATTO)));

    visualize.DofValues(dof1);
    visualize.WriteVtuFile(filename + ".vtu");
  }

  /* Solve Helmholtz equation:
   *
   * (-l*M + K)u = 0, with l = omega^2
   *
   * numEVals: number of Eigenvalues starting with the smallest
   *
   */
  void Solve(int numEVals) {
    std::string typeOfVal = "SM"; // Smallest magnitude

    Eigen::SparseMatrix<double> M =
        Eigen::MatrixXd(massMxMod.asDiagonal()).sparseView();

    Eigen::ArpackGeneralizedSelfAdjointEigenSolver<Eigen::SparseMatrix<double>>
        eigSolver(stiffnessMxMod, M, numEVals, typeOfVal);

    if (eigSolver.info() != Eigen::Success) {
      throw Exception("ArpackEigensolver did not succeed!");
    }

    Eigen::VectorXd eVals = eigSolver.eigenvalues();
    Eigen::MatrixXd eVecs = eigSolver.eigenvectors();

    for (int i = 0; i < eVecs.cols(); i++) {
      femResult.head(numDofsJ) = eVecs.col(i);
      MergeNodeVals(femResult, mesh.NodesTotal(dof1));
      Plot("HelmholtzMode" + std::to_string(i));
      std::cout << "Nr: " << i
                << "  omega (mtiltiples of pi): " << sqrt(eVals[i]) / M_PI
                << std::endl;
    }
  }
};

/*  Solves Helmholtz equation on a Rectangle:
 *  Starting from wave equation
 *
 *  d2u/dt2 - Delta u = f
 *
 *  Considering solutions of the form u(x,t) = e^{- i omega t}U(x)
 *  and periodic forcing f = e^{-i omega t}F
 *
 *  leads to:
 *
 *  -omega^2 u - Delta u = F
 *
 *
 * */
int main(int argc, char *argv[]) {
  // only use an even number of elements! The center node is fixed
  TestProblem example1(20, 2);
  example1.Solve(10);
}
