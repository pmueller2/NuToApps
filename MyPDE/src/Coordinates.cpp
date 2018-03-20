#include "../../NuToHelpers/MyDifferentialOperators.h"
#include "CoordinateSystem.h"
#include "OrthonormalCoordinateSystem.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include <iostream>

using namespace NuTo;

void TestCoordinatesClasses() {
  std::cout << "Coordinates system stuff" << std::endl;

  Eigen::Matrix3d J;
  J << 1., 0.1, 0., //
      0., 1., 0.,   //
      0., 0., 1.;   //

  Eigen::Matrix3d K1;
  K1 << 1., 0., 0., //
      0., 2., 0.,   //
      0., 0., 3.;   //

  Eigen::Matrix3d K2;
  K2 << 2., 0., 0., //
      0., 3., 0.,   //
      0., 0., 4.;   //

  Eigen::Matrix3d K3;
  K3 << 4., 0., 0., //
      0., 5., 0.,   //
      0., 0., 6.;   //

  CoordinateSystem coos(J, {K1, K2, K3});

  std::cout << "Jacobian/Transformationmatrix\n";
  std::cout << coos.GetJ();
  std::cout << std::endl;

  std::cout << "Metric\n";
  std::cout << coos.GetMetric();
  std::cout << std::endl;

  std::cout << "Christoffelsymbols\n";
  std::cout << coos.GetChristoffelSymbols()[0] << "\n ------- \n";
  std::cout << coos.GetChristoffelSymbols()[1] << "\n ------- \n";
  std::cout << coos.GetChristoffelSymbols()[2];
  std::cout << std::endl;

  OrthonormalCoordinateSystem ocoos(J);

  std::cout << "Jacobian/Transformationmatrix\n";
  std::cout << ocoos.GetJ();
  std::cout << std::endl;

  std::cout << "Metric\n";
  std::cout << ocoos.GetMetric();
  std::cout << std::endl;

  std::cout << "Christoffelsymbols\n";
  std::cout << ocoos.GetChristoffelSymbols()[0] << "\n ------- \n";
  std::cout << ocoos.GetChristoffelSymbols()[1] << "\n ------- \n";
  std::cout << ocoos.GetChristoffelSymbols()[2];
  std::cout << std::endl;
}

void TestMultiGradient() {

  int numNodes = 4;
  int dim = 2;
  int numDof = 3;

  Eigen::MatrixXd dNdX(numNodes, dim);
  dNdX << 1, 1.1, 2, 2.2, 3, 3.3, 4, 4.4;

  Nabla::MultiComponentGradient B(numDof);
  std::cout << "MulticomponentB:\n" << B(dNdX) << std::endl;
}

void TestCurl3D() {

  int numNodes = 4;
  int dim = 3;
  int numDof = 3;

  Eigen::MatrixXd dNdX(numNodes, dim);
  dNdX << 1, 1.1, 1.11, 2, 2.2, 2.22, 3, 3.3, 3.33, 4, 4.4, 4.44;

  Nabla::Curl3D B;
  std::cout << "Curl3D:\n" << B(dNdX) << std::endl;
}

void TestCurl2D() {

  int numNodes = 4;
  int dim = 2;
  int numDof = 2;

  Eigen::MatrixXd dNdX(numNodes, dim);
  dNdX << 1, 1.1, 2, 2.2, 3, 3.3, 4, 4.4;

  Nabla::Curl2DScalar B;
  std::cout << "Curl2DScalar:\n" << B(dNdX) << std::endl;
}

void TestVectorGradient() {
  Eigen::Matrix3d J;
  J << 1., 0.1, 0., //
      0., 1., 0.,   //
      0., 0., 1.;   //

  Eigen::Matrix3d K1;
  K1 << 1., 0., 0., //
      0., 2., 0.,   //
      0., 0., 3.;   //

  Eigen::Matrix3d K2;
  K2 << 2., 0., 0., //
      0., 3., 0.,   //
      0., 0., 4.;   //

  Eigen::Matrix3d K3;
  K3 << 4., 0., 0., //
      0., 5., 0.,   //
      0., 0., 6.;   //

  CoordinateSystem coos(J, {K1, K2, K3});

  int numNodes = 4;
  int dim = 3;

  Eigen::MatrixXd dNdX(numNodes, dim);
  dNdX << 1, 1.1, 1.11, 2, 2.2, 2.22, 3, 3.3, 3.33, 4, 4.4, 4.44;

  Nabla::VectorGradientCOOS B(coos);
  std::cout << "VectorGradientCOOS:\n" << B(dNdX) << std::endl;
}

int main(int argc, char *argv[]) {
  TestCoordinatesClasses();
  TestMultiGradient();
  TestCurl3D();
  TestCurl2D();
  TestVectorGradient();
}
