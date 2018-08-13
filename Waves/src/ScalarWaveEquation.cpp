#include "ScalarWaveEquation.h"
#include "../../NuToHelpers/ConstraintsHelper.h"
#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/integrationtypes/IntegrationCompanion.h"
#include "nuto/mechanics/interpolation/InterpolationCompanion.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"

using namespace NuTo;

ScalarWaveEquation::ScalarWaveEquation(MeshFem &mesh)
    : mDof("Scalar", 1), mMesh(mesh) {}

void ScalarWaveEquation::SetDomain(Group<ElementCollectionFem> elements) {
  mDomain = elements;
}

void ScalarWaveEquation::SetBoundary(Group<ElementCollectionFem> elements) {
  mBoundary = elements;
}

void ScalarWaveEquation::SetDirichletBoundary(
    Group<ElementCollectionFem> elements,
    std::function<double(Eigen::VectorXd)> func) {

  auto constraintEquations =
      NuTo::Constraint::SetDirichletBoundaryNodes(mDof, elements, func);

  mConstraints.Add(mDof, constraintEquations);
}

void ScalarWaveEquation::SetDirichletBoundary(
    NuTo::Group<NuTo::ElementCollectionFem> elements, double val) {
  SetDirichletBoundary(elements, [val](Eigen::VectorXd) { return val; });
}

void ScalarWaveEquation::SetDirichletBoundary(
    Group<NodeSimple> coordinateNodes,
    std::function<double(Eigen::VectorXd)> func) {

  for (NodeSimple &nd : coordinateNodes) {
    Eigen::VectorXd coords = nd.GetValues();
    NodeSimple &dofNode = mMesh.NodeAtCoordinate(coords, mDof);
    mConstraints.Add(mDof, Constraint::Value(dofNode, func(coords)));
  }
}

void ScalarWaveEquation::SetDirichletBoundary(Group<NodeSimple> coordinateNodes,
                                              double val) {
  SetDirichletBoundary(coordinateNodes, [val](Eigen::VectorXd) { return val; });
}

void ScalarWaveEquation::SetNeumannBoundary(
    NuTo::Group<NuTo::ElementCollectionFem> elements) {
  mNeumannBoundary = elements;
}

void ScalarWaveEquation::SetOrder(int order) {
  mInterpolationOrder = order;
  // ASSUME ALL SHAPES IN ELEMENTS ARE EQUAL (CHECK THIS)

  // create interpolation, add dof elements
  // *** Domain ***
  auto &ipolDomain = mMesh.CreateInterpolation(*CreateLobattoInterpolation(
      mDomain.begin()->CoordinateElement().GetShape(), order));
  AddDofInterpolation(&mMesh, mDof, mDomain, ipolDomain);

  // *** Boundaries ***
  if (!mBoundary.Empty()) {
    auto &ipolBoundary = mMesh.CreateInterpolation(*CreateLobattoInterpolation(
        mBoundary.begin()->CoordinateElement().GetShape(), order));
    AddDofInterpolation(&mMesh, mDof, mBoundary, ipolBoundary);
  }

  // create integration, add cells
  int integrationOrder = order + 1;

  // *** Domain ***
  mIntTypes.push_back(CreateLobattoIntegrationType(
      mDomain.begin()->DofElement(mDof).GetShape(), integrationOrder));

  int cellId = 0;
  for (auto &element : mDomain) {
    mCells.push_back(new NuTo::Cell(element, *mIntTypes.back(), cellId));
    cellId++;
    mDomainCells.Add(mCells.back());
  }

  // *** NeumannBoundary ***
  if (!mNeumannBoundary.Empty()) {
    mIntTypes.push_back(CreateLobattoIntegrationType(
        mNeumannBoundary.begin()->DofElement(mDof).GetShape(),
        integrationOrder));

    for (auto &element : mNeumannBoundary) {
      mCells.push_back(new NuTo::Cell(element, *mIntTypes.back(), cellId));
      cellId++;
      mNeumannBoundaryCells.Add(mCells.back());
    }
  }

  mDofNodes = mMesh.NodesTotal(mDof);
}

void ScalarWaveEquation::SetResultDirectory(std::string resultDirectory,
                                            bool overwriteResultDirectory) {
  // delete result directory if it exists and create it new
  boost::filesystem::path rootPath = boost::filesystem::initial_path();
  mResultDirectoryFull = rootPath.parent_path()
                             .parent_path()
                             .append("/results")
                             .append(resultDirectory);

  if (boost::filesystem::exists(mResultDirectoryFull)) // does p actually exist?
  {
    if (boost::filesystem::is_directory(mResultDirectoryFull)) {
      if (overwriteResultDirectory) {
        boost::filesystem::remove_all(mResultDirectoryFull);
        boost::filesystem::create_directory(mResultDirectoryFull);
      }
    }
  } else {
    boost::filesystem::create_directory(mResultDirectoryFull);
  }
}

void ScalarWaveEquation::Solve(double simTime, double timeStep) {

  // Set up assembler, dof nums, etc.
  DofInfo dofInfo = DofNumbering::Build(mDofNodes, mDof, mConstraints);
  int numDofsJ = dofInfo.numIndependentDofs[mDof];
  int numDofsK = dofInfo.numDependentDofs[mDof];
  int numDofs = numDofsJ + numDofsK;
  auto cmat = mConstraints.BuildUnitConstraintMatrix(mDof, numDofs);
  Eigen::PermutationMatrix<Eigen::Dynamic> JKtoGlobal(
      mConstraints.GetJKNumbering(mDof, numDofs).mIndices);
  Eigen::PermutationMatrix<Eigen::Dynamic> GlobalToJK = JKtoGlobal.inverse();

  SimpleAssembler asmbl = SimpleAssembler(dofInfo);

  // Compute mass matrix
  Eigen::VectorXd lumpedMassMx = asmbl.BuildDiagonallyLumpedMatrix(
      mDomainCells, {mDof}, [&](const CellIpData &cipd) {
        Eigen::MatrixXd N = cipd.N(mDof);
        DofMatrix<double> massLocal;
        massLocal(mDof, mDof) = N.transpose() * N;
        return massLocal;
      })[mDof];

  Eigen::VectorXd modMass =
      (cmat.transpose() * lumpedMassMx.asDiagonal() * cmat).eval().diagonal();

  // Compute stiffness matrix
  Eigen::SparseMatrix<double> stiffness =
      asmbl.BuildMatrix(mDomainCells, {mDof}, [&](const CellIpData &cipd) {
        Eigen::MatrixXd B = cipd.B(mDof, Nabla::Gradient());
        DofMatrix<double> stiffnessLocal;
        stiffnessLocal(mDof, mDof) = B.transpose() * B;
        return stiffnessLocal;
      })(mDof, mDof);

  Eigen::SparseMatrix<double> modStiffness =
      (cmat.transpose() * stiffness * cmat).eval();
}
