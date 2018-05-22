#include <iostream>

#include "nuto/mechanics/interpolation/InterpolationQuadLinear.h"
#include "nuto/mechanics/interpolation/InterpolationQuadSerendipity.h"
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

#include <eigen3/unsupported/Eigen/ArpackSupport>

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
      B(2, iColumn) = mN(1, iColumn + 1);
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
  double E = 210e9;

  int numElms = 1;
  int order = 10;

  Eigen::Matrix3d stiffnessTensor;
  stiffnessTensor << 1 - nu, nu, 0, //
      nu, 1 - nu, 0,                //
      0, 0, (1. - 2. * nu) / 2.;    //
  stiffnessTensor *= E / ((1. + nu) * (1 - 2. * nu));

  MeshFem mesh =
      UnitMeshFem::Transform(UnitMeshFem::CreateLines(numElms),
                             [](Eigen::VectorXd x) { return x * 0.01; });

  DofType dof("Displacements", 2);

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

  DofMatrixSparse<double> massMx =
      asmbl.BuildMatrix(cellGroup, {dof}, massMxFunc);

  auto E0MxFunc = [stiffnessTensor, dof](const CellIpData &cipd) {

    DofMatrix<double> E0Mx;
    Eigen::MatrixXd N = cipd.N(dof);
    Eigen::MatrixXd B1 = cipd.B(dof, Nabla::StrainB1(N));

    E0Mx(dof, dof) = B1.transpose() * stiffnessTensor * B1;
    return E0Mx;
  };

  auto E1MxFunc = [stiffnessTensor, dof](const CellIpData &cipd) {

    DofMatrix<double> E1Mx;
    Eigen::MatrixXd N = cipd.N(dof);

    //    Eigen::MatrixXd B1 = cipd.B(dof, Nabla::StrainB1(N));
    //    Eigen::MatrixXd B2 = cipd.B(dof, Nabla::StrainB2());

    Eigen::MatrixXd B1 =
        Nabla::StrainB1(N)(cipd.CalculateDerivativeShapeFunctionsGlobal(dof));
    Eigen::MatrixXd B2 =
        Nabla::StrainB2()(cipd.CalculateDerivativeShapeFunctionsGlobal(dof));

    E1Mx(dof, dof) = B2.transpose() * stiffnessTensor * B1;
    return E1Mx;
  };

  auto E2MxFunc = [stiffnessTensor, dof](const CellIpData &cipd) {

    DofMatrix<double> E2Mx;
    Eigen::MatrixXd B2 = cipd.B(dof, Nabla::StrainB2());

    E2Mx(dof, dof) = B2.transpose() * stiffnessTensor * B2;
    return E2Mx;
  };

  Eigen::SparseMatrix<double> E0Mx =
      asmbl.BuildMatrix(cellGroup, {dof}, E0MxFunc)(dof, dof);
  Eigen::SparseMatrix<double> E1Mx =
      asmbl.BuildMatrix(cellGroup, {dof}, E1MxFunc)(dof, dof);
  Eigen::SparseMatrix<double> E2Mx =
      asmbl.BuildMatrix(cellGroup, {dof}, E2MxFunc)(dof, dof);

  Eigen::SparseMatrix<double> M = massMx(dof, dof);

  /*
  std::cout << "E0\n" << E0Mx.toDense() << std::endl;
  std::cout << "E1\n" << E1Mx.toDense() << std::endl;
  std::cout << "E2\n" << E2Mx.toDense() << std::endl;
  std::cout << "M\n" << M.toDense() << std::endl;
  */

  // ************************
  // Construct Z(omega)
  // ************************

  int numOmega = 1000;
  double omegaMax = 1e6;

  for (int i = 1; i < numOmega; i++) {
    double omega = i * omegaMax / (numOmega - 1);
    // double omega = 0.55e6;

    Eigen::MatrixXd Z11 = E0Mx.toDense().inverse() * E1Mx.toDense().transpose();
    Eigen::MatrixXd Z12 = -E0Mx.toDense().inverse();
    Eigen::MatrixXd Z21 = omega * omega * M - E2Mx +
                          E1Mx * (E0Mx.toDense().inverse() * E1Mx.transpose());
    Eigen::MatrixXd Z22 = -E1Mx * E0Mx.toDense().inverse();

    Eigen::MatrixXd Z(Z11.rows() + Z21.rows(), Z11.cols() + Z12.cols());
    Z << Z11, Z12, Z21, Z22;

    Eigen::MatrixXd Zexpected(Z11.rows() + Z21.rows(), Z11.cols() + Z12.cols());

    //    Zexpected << -0.0000000e+00, -4.2857143e+01, -0.0000000e+00,
    //    4.2857143e+01,
    //        -7.0748299e-10, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    //        -1.0000000e+02, -0.0000000e+00, 1.0000000e+02, -0.0000000e+00,
    //        0.0000000e+00, -2.4761905e-09, 0.0000000e+00, 0.0000000e+00,
    //        -0.0000000e+00, -4.2857143e+01, -0.0000000e+00, 4.2857143e+01,
    //        0.0000000e+00, 0.0000000e+00, -7.0748299e-10, 0.0000000e+00,
    //        -1.0000000e+02, -0.0000000e+00, 1.0000000e+02, -0.0000000e+00,
    //        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, -2.4761905e-09,
    //        9.7656250e-04, -0.0000000e+00, -9.7656250e-04, -0.0000000e+00,
    //        0.0000000e+00, 1.0000000e+02, 0.0000000e+00, 1.0000000e+02,
    //        -0.0000000e+00, -2.3076923e+13, -0.0000000e+00, 2.3076923e+13,
    //        4.2857143e+01, 0.0000000e+00, 4.2857143e+01, 0.0000000e+00,
    //        -9.7656250e-04, -0.0000000e+00, 9.7656250e-04, -0.0000000e+00,
    //        0.0000000e+00, -1.0000000e+02, 0.0000000e+00, -1.0000000e+02,
    //        -0.0000000e+00, 2.3076923e+13, -0.0000000e+00, -2.3076923e+13,
    //        -4.2857143e+01, 0.0000000e+00, -4.2857143e+01, 0.0000000e+00;

    //    std::cout << Z << std::endl;
    //    std::cout << "-----------------" << std::endl;
    //    std::cout << Zexpected << std::endl;

    // ************************
    // Solve Eigenvalue problem
    // ************************

    Eigen::EigenSolver<Eigen::MatrixXd> solver(Z);
    Eigen::VectorXcd eVals = solver.eigenvalues();

    // std::cout << eVals << std::endl;

    std::vector<double> eValCandidates;

    for (int i = 0; i < eVals.size(); i++) {
      std::complex<double> k = eVals[i];
      if ((std::abs(k.real()) < 10.) && (k.imag() > 0.))
        eValCandidates.push_back(k.imag());
    }

    std::sort(eValCandidates.begin(), eValCandidates.end());

    std::cout << omega << " ";
    for (auto k : eValCandidates) {
      std::cout << "\t" << sqrt(k);
    }
    std::cout << std::endl;
  }

  //  Visualize::Visualizer vis(cellGroup, Visualize::AverageHandler());
  //  vis.DofValues(dof);
  //  vis.WriteVtuFile("Test.vtu");
}
