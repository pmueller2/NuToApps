#include <iostream>

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationQuadQuadratic.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/mechanics/cell/Cell.h"

#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/Visualizer.h"

namespace NuTo {

namespace Nabla {

/* dNdX - shape function derivatives (nodes, spaceDim)
 *
 *
*/
struct StrainK2D : Interface {

  StrainK2D(double k, const Eigen::MatrixXd N) : mK(k), mN(N) {}

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &dNdX) const override {
    const int dim = dNdX.cols();
    assert(dim == 2);
    const int numNodes = dNdX.rows();

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, numNodes * 2);
    for (int iNode = 0, iColumn = 0; iNode < numNodes; ++iNode, iColumn += 2) {
      double dNdXx = dNdX(iNode, 0);
      B(0, iColumn) = dNdXx;
      B(1, iColumn + 1) = mK * mN(iNode);
      B(2, iColumn) = mK * mN(iNode);
      B(2, iColumn + 1) = dNdXx;
    }
    return B;
  }

  double mK;
  const Eigen::MatrixXd &mN;
};

struct StrainB1 : Interface {

  StrainB1(const Eigen::MatrixXd N) : mN(N) {}

  Eigen::MatrixXd operator()(const Eigen::MatrixXd & /*dNdX */) const override {
    const int numNodes = mN.cols() / 2;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, numNodes * 2);
    for (int iNode = 0, iColumn = 0; iNode < numNodes; ++iNode, iColumn += 2) {
      B(1, iColumn + 1) = mN(0, iColumn);
      B(2, iColumn) = mN(0, iColumn);
    }

    return B;
  }

  const Eigen::MatrixXd &mN;
};

struct StrainB2 : Interface {

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &dNdX) const override {
    const int dim = dNdX.cols();
    const int numNodes = dNdX.rows();

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, numNodes * 2);
    for (int iNode = 0, iColumn = 0; iNode < numNodes; ++iNode, iColumn += 2) {
      double dNdXx = dNdX(iNode, 0);
      B(0, iColumn) = dNdXx;
      B(2, iColumn + 1) = dNdXx;
    }
    return B;
  }
};

} /* B */
} /* NuTo */

using namespace NuTo;

int main(int argc, char *argv[]) {

  // **********************
  // Material parameters
  // **********************

  double rho = 7850;
  double nu = 0.3;
  double E = 210;

  Eigen::Matrix3d stiffnessTensor;
  stiffnessTensor << 1 - nu, nu, 0, //
      nu, 1 - nu, 0,                //
      0, 0, (1. - 2. * nu) / 2.;    //
  stiffnessTensor *= E / ((1 + nu) * (1 - 2 * nu));

  MeshFem mesh = UnitMeshFem::CreateLines(1);

  DofType dof("Displacements", 2);

  int order = 10;

  InterpolationTrussLobatto ipol(order);

  AddDofInterpolation(&mesh, dof, ipol);

  IntegrationTypeTensorProduct<1> integr(order + 1,
                                         eIntegrationMethod::LOBATTO);

  CellStorage cells;
  Group<CellInterface> cellGroup = cells.AddCells(mesh.ElementsTotal(), integr);

  Constraint::Constraints constraints;

  DofInfo dofInfo = DofNumbering::Build(mesh.NodesTotal(dof), dof, constraints);

  // **********************
  // Matrix assembly
  // **********************

  SimpleAssembler asmbl(dofInfo);

  auto massMxFunc = [&](const CellIpData &cipd) {

    Eigen::MatrixXd N = cipd.N(dof);
    DofMatrix<double> massMx;

    massMx(dof, dof) = N.transpose() * N * rho;
    return massMx;
  };

  GlobalDofMatrixSparse massMx =
      asmbl.BuildMatrix(cellGroup, {dof}, massMxFunc);

  auto E0MxFunc = [&](const CellIpData &cipd) {

    DofMatrix<double> E0Mx;
    Eigen::MatrixXd N = cipd.N(dof);
    Eigen::MatrixXd B1 = cipd.B(dof, Nabla::StrainB1(N));

    E0Mx(dof, dof) = B1.transpose() * stiffnessTensor * B1;
    return E0Mx;
  };

  auto E1MxFunc = [&](const CellIpData &cipd) {

    DofMatrix<double> E1Mx;
    Eigen::MatrixXd N = cipd.N(dof);
    Eigen::MatrixXd B1 = cipd.B(dof, Nabla::StrainB1(N));
    Eigen::MatrixXd B2 = cipd.B(dof, Nabla::StrainB2());

    E1Mx(dof, dof) = B2.transpose() * stiffnessTensor * B1;
    return E1Mx;
  };

  auto E2MxFunc = [&](const CellIpData &cipd) {

    DofMatrix<double> E2Mx;
    Eigen::MatrixXd B2 = cipd.B(dof, Nabla::StrainB2());

    E2Mx(dof, dof) = B2.transpose() * stiffnessTensor * B2;
    return E2Mx;
  };

  Eigen::SparseMatrix<double> E0Mx =
      asmbl.BuildMatrix(cellGroup, {dof}, E0MxFunc).JJ(dof, dof);
  Eigen::SparseMatrix<double> E1Mx =
      asmbl.BuildMatrix(cellGroup, {dof}, E1MxFunc).JJ(dof, dof);
  Eigen::SparseMatrix<double> E2Mx =
      asmbl.BuildMatrix(cellGroup, {dof}, E2MxFunc).JJ(dof, dof);

  Eigen::SparseMatrix<double> M = massMx.JJ(dof, dof);

  // ************************
  // Construct Z(omega)
  // ************************

  int numOmega = 100;
  double omegaMax = 1e6;

  for (int i = 0; i < numOmega; i++) {
    double omega = i * omegaMax / (numOmega - 1);

    Eigen::MatrixXd Z11 = E0Mx.toDense().inverse() * E1Mx.toDense().transpose();
    Eigen::MatrixXd Z12 = -E0Mx.toDense().inverse();
    Eigen::MatrixXd Z21 = omega * omega * M - E2Mx +
                          E1Mx * E0Mx.toDense().inverse() * E0Mx.transpose();
    Eigen::MatrixXd Z22 = -E1Mx * E0Mx.toDense().inverse();

    Eigen::MatrixXd Z(Z11.rows() + Z21.rows(), Z11.cols() + Z12.cols());
    Z << Z11, Z12, Z21, Z22;

    // ************************
    // Solve Eigenvalue problem
    // ************************

    Eigen::EigenSolver<Eigen::MatrixXd> solver(Z);
    Eigen::VectorXcd eVals = solver.eigenvalues();
    Eigen::VectorXd k = eVals.imag();
    int n = k.size();

    for (int i = 0; i < n; i++) {
      if (k[i] < 1.e-6)
        k[i] = INFINITY;
    }

    std::sort(k.data(), k.data() + k.size());

    std::cout << omega << " " << k[0] << "\t" << k[1] << "\t" << k[2]
              << std::endl;
  }

  //  Visualize::Visualizer vis(cellGroup, Visualize::AverageHandler());
  //  vis.DofValues(dof);
  //  vis.WriteVtuFile("Test.vtu");
}
