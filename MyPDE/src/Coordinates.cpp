#include "../../NuToHelpers/MyDifferentialOperators.h"
#include "CoordinateSystem.h"
#include "CylindricalCoordinates.h"
#include "OrthonormalCoordinateSystem.h"
#include "SphericalCoordinates.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/tools/CellStorage.h"
#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/Visualizer.h"

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

void TestLinearSystem() {
  Eigen::Vector3d a(1., 0., 0.);
  Eigen::Vector3d b(0., 2., 0.);
  Eigen::Vector3d c(0., 0., 3.);

  Eigen::Matrix3d jac;
  jac << a, b, c;

  std::cout << jac << std::endl;

  CoordinateSystem cs(jac);

  Eigen::Vector3d v(1, 1, 1);

  std::cout << "Test Transformations: " << std::endl;
  std::cout << "Testvector is (1,1,1):\n " << std::endl;
  std::cout << "C to T (1,1/2,1/3)\n" << cs.TransformVectorCtoT(v) << std::endl;
  std::cout << "C to G (1,2,3)   \n " << cs.TransformVectorCtoG(v) << std::endl;
  std::cout << "T to C (1,2,3)  \n  " << cs.TransformVectorTtoC(v) << std::endl;
  std::cout << "G to C (1,1/2,1/3)\n" << cs.TransformVectorGtoC(v) << std::endl;
  std::cout << "G to T (1,1/4,1/9)\n" << cs.TransformVectorGtoT(v) << std::endl;
  std::cout << "T to G (1,4,9)\n" << cs.TransformVectorTtoG(v) << std::endl;
}

void TestCylindrical() {
  CylindricalCoordinates cs;

  Eigen::Vector3d v(10., M_PI, 1.);
  std::cout << cs.GetCartesian(v) << std::endl;

  CoordinateSystem coos = cs.GetNaturalCOOS(v);
  std::cout << coos.GetMetric() << std::endl;
}

void TestSpherical() {
  SphericalCoordinates cs;

  Eigen::Vector3d v(10., M_PI, 1.);
  std::cout << cs.GetCartesian(v) << std::endl;

  CoordinateSystem coos = cs.GetNaturalCOOS(v);
  std::cout << coos.GetMetric() << std::endl;
}

void TestSkewLinearSystem() {
  Eigen::Vector3d a(1., 0., 0.);
  Eigen::Vector3d b(0., 1., 0.);
  Eigen::Vector3d c(1., 1., 1.);

  Eigen::Matrix3d jac;
  jac << a, b, c;

  std::cout << jac << std::endl;

  CoordinateSystem cs(jac);

  Eigen::Vector3d v(0, 0, 1);

  std::cout << "Test Transformations: " << std::endl;
  std::cout << "Testvector is (0,0,1):\n " << std::endl;
  std::cout << "C to T \n" << cs.TransformVectorCtoT(v) << std::endl;
  std::cout << "C to G \n " << cs.TransformVectorCtoG(v) << std::endl;
  std::cout << "T to C \n  " << cs.TransformVectorTtoC(v) << std::endl;
  std::cout << "G to C \n" << cs.TransformVectorGtoC(v) << std::endl;
  std::cout << "G to T \n" << cs.TransformVectorGtoT(v) << std::endl;
  std::cout << "T to G \n" << cs.TransformVectorTtoG(v) << std::endl;
}

void TestScalarGradient() {

  auto fCartesian = [](Eigen::Vector3d coords) {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    return 0.1 * x * x + 0.3 * y * y + z * z;
  };

  auto dfCartesian = [](Eigen::Vector3d coords) {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    return Eigen::Vector3d(0.2 * x, 0.6 * y, 2. * z);
  };

  // Set up a mesh
  MeshFem m = UnitMeshFem::CreateBricks(10, 10, 10);
  DofType dof("scalar", 1);
  AddDofInterpolation(&m, dof);

  for (ElementCollectionFem &elm : m.Elements) {
    ElementFem cElm = elm.CoordinateElement();
    ElementFem dElm = elm.DofElement(dof);
    for (int i = 0; i < dElm.Interpolation().GetNumNodes(); i++) {
      Eigen::Vector3d coords =
          Interpolate(cElm, dElm.Interpolation().GetLocalCoords(i));
      double val = fCartesian(coords);
      dElm.GetNode(i).SetValue(0, val);
    }
  }

  IntegrationTypeTensorProduct<3> integrationType3D(1,
                                                    eIntegrationMethod::GAUSS);

  // volume cells
  CellStorage cells;
  Group<CellInterface> cellGroup =
      cells.AddCells(m.ElementsTotal(), integrationType3D);

  NuTo::Visualize::Visualizer visualize(cellGroup,
                                        NuTo::Visualize::AverageHandler());
  visualize.DofValues(dof);
  visualize.WriteVtuFile("CoordinatesTest.vtu");
}

int main(int argc, char *argv[]) {
  TestScalarGradient();
  // TestCylindrical();
  // TestSpherical();
  //  TestLinearSystem();
  //  TestSkewLinearSystem();
  //  TestCoordinatesClasses();
  //  TestMultiGradient();
  //  TestCurl3D();
  //  TestCurl2D();
  //  TestVectorGradient();
}
