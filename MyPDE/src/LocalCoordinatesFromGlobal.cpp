#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/interpolation/InterpolationQuadLobatto.h"
#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "nuto/mechanics/tools/CellStorage.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include <iostream>

using namespace NuTo;

/*                        4
 *                          1
 *
 *
 *            3
 *                       2
 *
 *  x = sum_i  x_i * Ni(X,Y)
 *  y          y_i
 *                                              NodeValsPhil^T x der         = Jac
 * dx/dX = sum_i  x_i * dNi/dX     ->  |x1, x2, x3, x4|  .   |dN1/dX dN1/dY| = |dx/dX  dx/dY|
 * dy/dX          y_i                  |y1, y2, y3, y4|      |dN2/dX dN2/dY|   |dy/dX  dy/dY|
 *                                                           |dN3/dX dN3/dY|
 * dx/dY = sum_i  x_i * dNi/dY                               |dN4/dX dN4/dY|
 * dy/dY          y_i
 *
 * Newton:  f(x) = 0
 *          f(x0) + (x-x0) f' = 0, dx = x-x0
 *           dx = solve(f',-f(x0)), x = x0+dx
 *
 *   Here: f(X)  =  x(X) - P, f' = Jac
 *
 *  ->      dX = solve(Jac, P -x(X0)), X = X0 + dX
 *
*/
void MyTest()
{
    NodeSimple nd1(Eigen::Vector2d(4.,5.1));
    NodeSimple nd2(Eigen::Vector2d(3.,1.));
    NodeSimple nd3(Eigen::Vector2d(1.,2.));
    NodeSimple nd4(Eigen::Vector2d(3.5,6.));

    InterpolationQuadLobatto ipol(1);

    std::cout << "Created Nodes and interpolation" << std::endl;

    ElementFem elm({nd1,nd2,nd3,nd4},ipol);

    std::cout << "Created Quad Element" << std::endl;

    Eigen::Vector2d localTestCoord(0.2,0.3);
    Eigen::Vector2d globalTestCoord = Interpolate(elm,localTestCoord);

    std::cout << "Test Coord local : " << localTestCoord[0] << ", " << localTestCoord[1] << std::endl;
    std::cout << "   ->     global : " << globalTestCoord[0] << ", " << globalTestCoord[1] << std::endl;

    std::cout << "Attempt reverse mapping using newtons method." << std::endl;
    Eigen::Vector2d xi(0.0,0.0);
    std::cout << "Initial guess: " << xi[0] << ", " << xi[1] << std::endl;

    auto der = ipol.GetDerivativeShapeFunctions(xi);
    int spaceDim = 2;
    Eigen::MatrixXd nodeVals(ipol.GetNumNodes(),spaceDim);
    for (int i=0; i<nodeVals.rows();i++)
    {
        nodeVals.row(i) = elm.GetNode(i).GetValues().transpose();
    }
    std::cout << "Shape function derivative (nodes x spaceDim): " << der << std::endl;

    std::cout << "NodalValues NutoStyle: " << elm.ExtractNodeValues() << std::endl;
    std::cout << "NodalValues Phil     : " << nodeVals << std::endl;

    Eigen::Matrix2d jac = nodeVals.transpose() * der;
    std::cout << "Jacobian: " << jac << std::endl;

    std::cout << "Sart Newton iterations " << std::endl;

    for (int i=0; i<10; i++)
    {
        der = ipol.GetDerivativeShapeFunctions(xi);
        jac = nodeVals.transpose() * der;
        Eigen::Vector2d res = globalTestCoord - Interpolate(elm,xi);
        Eigen::Vector2d dX = jac.inverse() * res;
        Eigen::Vector2d xiNew = xi + dX;
        xi = xiNew;
        std::cout << i << " " << xi[0] << ", " << xi[1] << std::endl;
    }
}

int main(int argc, char *argv[]) {
   MeshGmsh gmsh("plateWithInternalCrackHexed.msh");
   MeshFem &mesh = gmsh.GetMeshFEM();
   auto domain = gmsh.GetPhysicalGroup("Domain");

   for (ElementCollectionFem& elmColl : domain)
   {
       ElementFem& elm = elmColl.CoordinateElement();
       auto& ipol = elm.Interpolation();
       int spaceDim = 3;
       Eigen::Vector3d localTestCoord(1.1,-1.2,1.4);
       Eigen::VectorXd globalTestCoord = Interpolate(elm,localTestCoord);
       Eigen::MatrixXd nodeVals(ipol.GetNumNodes(),spaceDim);
       for (int i=0; i<nodeVals.rows();i++)
       {
           nodeVals.row(i) = elm.GetNode(i).GetValues().transpose();
       }
       Eigen::VectorXd xi = Eigen::VectorXd::Zero(spaceDim);
       int numIterations = 0;
       double correctorNorm = 1.;
       int maxNumIterations = 20;
       double tol = 1e-7;
       while ((correctorNorm > tol) && (numIterations < maxNumIterations))
       {
           Eigen::MatrixXd der = ipol.GetDerivativeShapeFunctions(xi);
           Eigen::MatrixXd jac = nodeVals.transpose() * der;
           Eigen::VectorXd res = globalTestCoord - Interpolate(elm,xi);
           Eigen::VectorXd dX = jac.inverse() * res;
           correctorNorm = dX.norm();
           Eigen::VectorXd xiNew = xi + dX;
           xi = xiNew;
           numIterations++;
       }
       std::cout << numIterations << " ";
       std::cout << correctorNorm << " ";
       std::cout << xi[0] << ", " << xi[1] << ", " << xi[2] << std::endl;
       std::cout << "------------------------" << std::endl;
   }
}
