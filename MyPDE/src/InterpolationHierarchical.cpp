#include <iostream>

#include "../../NuToHelpers/InterpolationQuadHierarchical.h"
#include "../../NuToHelpers/InterpolationTrussHierarchical.h"

using namespace NuTo;

void TestHTruss() {
  InterpolationTrussHierarchical ipol(5);
  int n = 100;
  for (int i = 0; i < n; i++) {
    double x = -1. + i * 2. / (n - 1);
    Eigen::VectorXd shapes =
        ipol.GetShapeFunctions(Eigen::VectorXd::Constant(1, x));
    for (int j = 0; j < shapes.size(); j++) {
      std::cout << shapes[j] << "\t";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) { TestHTruss(); }
