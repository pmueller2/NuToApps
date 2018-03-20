#include <iostream>

#include "mechanics/mesh/MeshFem.h"

#include "mechanics/cell/Cell.h"
#include "mechanics/integrationtypes/IntegrationType2D3NGauss4Ip.h"
#include "mechanics/interpolation/InterpolationTriangleLinear.h"

using namespace NuTo;

/* Generate a mesh and try to access all relevant geometry information
 * Starting with surface normals
 */
int main(int argc, char *argv[]) {
  MeshFem mesh;
  NodeSimple nd0(Eigen::Vector3d(1., 0., 0.));
  NodeSimple nd1(Eigen::Vector3d(0., 1., 0.));
  NodeSimple nd2(Eigen::Vector3d(0., 0., 1.));

  int dim = nd0.GetValues().size();

  auto &ipol = mesh.CreateInterpolation(InterpolationTriangleLinear());
  mesh.Elements.Add({{{nd0, nd1, nd2}, ipol}});

  auto integr = IntegrationType2D3NGauss4Ip();
  int cellId = 0;
  auto c = new Cell(mesh.Elements[0], integr, cellId);
  Group<CellInterface> cells;
  cells.Add(*c);

  DofType dof("dof", 1);

  CellData cellData(mesh.Elements[0], cellId);

  // Get geometry information from boundaries
  std::vector<Eigen::Vector3d> ipCoordVector = {
      Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(0.2, 0., 0.),
      Eigen::Vector3d(0.2, 0.2, 0.), Eigen::Vector3d(0.5, 0.5, 0.5)};

  for (Eigen::Vector3d ipCoords : ipCoordVector) {
    auto &cElm = mesh.Elements[0].CoordinateElement();

    Eigen::MatrixXd ShapesDerivative =
        cElm.GetDerivativeShapeFunctions(ipCoords);
    Eigen::VectorXd nodeVals = cElm.ExtractNodeValues();

    Eigen::MatrixXd nodeBlockCoordinates(ShapesDerivative.rows(), dim);

    for (int i = 0; i < ShapesDerivative.rows(); i++) {
      nodeBlockCoordinates(i, 0) = nodeVals(3 * i);
      nodeBlockCoordinates(i, 1) = nodeVals(3 * i + 1);
      nodeBlockCoordinates(i, 2) = nodeVals(3 * i + 2);
    }

    Eigen::Matrix<double, 3, 2> result =
        nodeBlockCoordinates.transpose() * ShapesDerivative;
    Eigen::Vector3d normal = (result.col(0)).cross(result.col(1));

    double nrm = normal.norm();
    if (nrm != 0)
      normal /= nrm;
    std::cout << "Self computed ";
    std::cout << normal[0] << "   " << normal[1] << "   " << normal[2]
              << std::endl;

    Jacobian jc(nodeVals, ShapesDerivative, 3);
    auto newNormal = jc.Get().col(2);
    std::cout << newNormal[0] << "   " << newNormal[1] << "   " << newNormal[2]
              << std::endl;
  }
}
