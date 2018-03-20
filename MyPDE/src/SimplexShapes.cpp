#include "MyLagrangeShapes.h"
#include <iostream>
#include <vector>

std::vector<double> Linspace(double s, double e, size_t num) {
  std::vector<double> linspace(num);
  for (size_t i = 0; i < num; ++i)
    linspace[i] = s + (e - s) * i / (num - 1);
  return linspace;
}

void TestTriangle(int order) {
  std::vector<double> partition = Linspace(0., 1., order + 1);

  for (double x : partition)
    for (double y : partition) {
      if (x + y > 1)
        continue;
      Eigen::VectorXd shapes =
          NuTo::ShapeFunctions2D::ShapeFunctionsTriangleLagrange(
              Eigen::Vector2d(x, y), partition);
      for (int i = 0; i < shapes.size(); ++i)
        std::cout << std::to_string(shapes[i]) << " ";
      std::cout << std::endl;
    }
}

void TestTetrahedron(int order) {
  std::vector<double> partition = Linspace(0., 1., order + 1);

  for (double x : partition)
    for (double y : partition)
      for (double z : partition) {
        if (x + y + z > 1)
          continue;
        Eigen::VectorXd shapes =
            NuTo::ShapeFunctions3D::ShapeFunctionsTetrahedronLagrange(
                Eigen::Vector3d(x, y, z), partition);
        for (int i = 0; i < shapes.size(); ++i)
          std::cout << std::to_string(shapes[i]) << " ";
        std::cout << std::endl;
      }
}

int main(int argc, char *argv[]) {
  std::cout << "Lagrangian simplex shape functions of arbitrary order"
            << std::endl;

  // Create equidistant partition of [0,1]
  int order = 3;
  TestTriangle(order);
  TestTetrahedron(order);
}
