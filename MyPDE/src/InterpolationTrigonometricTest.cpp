#include <iostream>

#include "../../NuToHelpers/InterpolationTrussTrigonometric.h"

using namespace NuTo;

int main(int argc, char *argv[]) {

  const InterpolationTrussTrigonometric interpolation(6);

  int N = interpolation.GetNumNodes();

  int nOut = 100;

  for (int i = 0; i < nOut; i++) {
    double x = -1 + 2 * i / (nOut - 1.);
    Eigen::VectorXd shapes =
        interpolation.GetShapeFunctions(Eigen::VectorXd::Constant(1, x));
    std::cout << x << "   ";
    for (int j = 0; j < shapes.size(); j++)
      std::cout << shapes[j] << "   ";
    std::cout << std::endl;
  }

  // Test derivatives
  //  for (int i = 0; i < nOut; i++) {
  //    double x = -1 + 2 * i / (nOut - 1.);
  //    Eigen::VectorXd dshapes = interpolation.GetDerivativeShapeFunctions(
  //        Eigen::VectorXd::Constant(1, x));
  //    Eigen::VectorXd shapes1 =
  //        interpolation.GetShapeFunctions(Eigen::VectorXd::Constant(1, x));
  //    double dx = 1.e-8;
  //    Eigen::VectorXd shapes2 =
  //        interpolation.GetShapeFunctions(Eigen::VectorXd::Constant(1, x +
  //        dx));
  //    Eigen::VectorXd dshapes2 = (shapes2 - shapes1) / dx;

  //    std::cout << x << "   ";
  //    for (int j = 0; j < dshapes.size(); j++) {
  //      std::cout << dshapes[j] << "   ";
  //      std::cout << dshapes2[j] << "   ";
  //    }
  //    std::cout << std::endl;
  //  }
}
