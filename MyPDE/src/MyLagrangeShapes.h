#pragma once

#include <Eigen/Core>
#include <vector>

namespace NuTo
{

namespace ShapeFunctions2D
{
//! @brief Value of Lagrange type shape functions with given 1D nodes
//! @param x local coordinate where shapes are evaluated
//! @param nodes local node coordinates
//! @return shapes evaluated at x. Fulfill interpolation condition fi(xj) = delta_ij
Eigen::VectorXd ShapeFunctionsTriangleLagrange(const Eigen::Vector2d x, const std::vector<double>& partition)
{
    //Create barycentric coordinates from local coordinates
    //after that work only with these
    double x1 = x[0];
    double x2 = x[1];
    double x3 = 1-x1-x2;

    int numNodes = partition.size();

    Eigen::VectorXd result((numNodes*(numNodes+1))/2);

    int count = 0;
    for (int k=numNodes-1; k>=0; --k)
        for (int l=numNodes-1-k; l>=0; --l)
        {
            int m = numNodes-1-k-l;
            double shape = 1.;
            for (int ik=0; ik<k; ++ik)
                shape *= (x1-partition[ik])/(partition[k]-partition[ik]);
            for (int il=0; il<l; ++il)
                shape *= (x2-partition[il])/(partition[l]-partition[il]);
            for (int im=0; im<m; ++im)
                shape *= (x3-partition[im])/(partition[m]-partition[im]);
            result[count] = shape;
            ++count;
        }
    return result;
}

Eigen::MatrixXd DerivativeShapeFunctionsTriangleLagrange(const Eigen::Vector2d x, const std::vector<double>& partition)
{
    Eigen::MatrixXd resultBarycentric;
    return resultBarycentric;
}

}


namespace ShapeFunctions3D
{

//! @brief Value of Lagrange type shape functions with given 1D nodes
//! @param x local coordinate where shapes are evaluated
//! @param nodes local node coordinates
//! @return shapes evaluated at x. Fulfill interpolation condition fi(xj) = delta_ij
Eigen::VectorXd ShapeFunctionsTetrahedronLagrange(const Eigen::Vector3d x, const std::vector<double>& partition)
{
    //Create barycentric coordinates from local coordinates
    //after that work only with these
    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];
    double x4 = 1-x1-x2-x3;

    int numNodes = partition.size();

    Eigen::VectorXd result((numNodes*(numNodes+1)*(numNodes+2))/6);

    int count = 0;
    for (int k=numNodes-1; k>=0; --k)
        for (int l=numNodes-1-k; l>=0; --l)
            for (int m=numNodes-1-k-l; m>=0; --m)
            {
                int n = numNodes-1-k-l-m;
                double shape = 1.;
                for (int ik=0; ik<k; ++ik)
                    shape *= (x1-partition[ik])/(partition[k]-partition[ik]);
                for (int il=0; il<l; ++il)
                    shape *= (x2-partition[il])/(partition[l]-partition[il]);
                for (int im=0; im<m; ++im)
                    shape *= (x3-partition[im])/(partition[m]-partition[im]);
                for (int iN=0; iN<n; ++iN)
                    shape *= (x4-partition[iN])/(partition[n]-partition[iN]);
                result[count] = shape;
                ++count;
            }
    return result;
}

Eigen::MatrixXd DerivativeShapeFunctionsTetrahedronLagrange(const Eigen::Vector3d x, const std::vector<double>& partition)
{
    Eigen::MatrixXd result;
    return result;
}

}

} /* namespace NuTo */
