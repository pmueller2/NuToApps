#pragma once

#include <Eigen/Core>
#include "nuto/base/Exception.h"
#include "nuto/mechanics/cell/DifferentialOperators.h"
#include "nuto/mechanics/DirectionEnum.h"

#include "../MyPDE/src/CoordinateSystem.h"

#include <vector>

namespace NuTo
{

namespace Nabla
{
//!@brief Gradient operator with some spatial variables replaced with wavenumber
struct GradientKSpace : Interface
{
    GradientKSpace(std::vector<double> k, std::vector<eDirection> d) : mK(k), mD(d)
    {

    }

    Eigen::MatrixXd operator()(const Eigen::MatrixXd& dNdX) const override
    {
        return dNdX.transpose();
    }

    std::vector<double> mK;
    std::vector<eDirection> mD;
};

//!@brief Curl operator for 3d dof
struct Curl3D : Interface
{
    //! @brief Curl of 3d dof
    //!
    //! curl v =
    //!  dvZ/dy - dvY/dz
    //!  dvX/dz - dvZ/dx
    //!  dvY/dx - dvX/dy
    //! =
    //!       0      - dvY/dz     dvZ/dy
    //!     dvX/dz      0       - dvZ/dx
    //!   - dvX/dy     dvY/dx        0
    //!
    //! B = |     0      - dvY/dz     dvZ/dy  ... |
    //!     |    dvX/dz      0       - dvZ/dx ... |
    //!     |  - dvX/dy    dvY/dx        0    ... |
    //!
    Eigen::MatrixXd operator()(const Eigen::MatrixXd& dNdX) const override
    {
        const int dim = dNdX.cols();
        if (dim != 3)
            throw Exception("Curl3D works only for 3 dim space");
        const int numNodes = dNdX.rows();
        Eigen::MatrixXd result(dim,numNodes*dim);
        for (int j=0;j<numNodes;j++)
        {
            result(0  , j*dim  ) = 0.;
            result(0  , j*dim+1) = -dNdX(j,2);
            result(0  , j*dim+2) =  dNdX(j,1);

            result(1, j*dim  ) =  dNdX(j,2);
            result(1, j*dim+1) = 0.;
            result(1, j*dim+2) = -dNdX(j,0);

            result(2, j*dim  ) = -dNdX(j,1);
            result(2, j*dim+1) =  dNdX(j,0);
            result(2, j*dim+2) = 0.;
        }
        return result;
    }
};

//!@brief Scalar Curl operator for 2d dof giving dFy/dx - dFx/dy
struct Curl2DScalar : Interface
{
    //! @brief Scalr Curl of 2d dof giving dFy/dx - dFx/dy
    //!
    //! B = |  -dFx/dy     +dFy/dx ... |
    //!
    Eigen::MatrixXd operator()(const Eigen::MatrixXd& dNdX) const override
    {
        const int dim = dNdX.cols();
        if (dim != 2)
            throw Exception("Curl2D works only for 2 dim space");
        const int numNodes = dNdX.rows();
        Eigen::MatrixXd result(1,numNodes*dim);
        for (int j=0;j<numNodes;j++)
        {
            result(0  , j*dim  ) = -dNdX(j,1);
            result(0  , j*dim+1) =  dNdX(j,0);
        }
        return result;
    }
};

//!@brief Gradient operator for multicomponent dof
struct MultiComponentGradient : Interface
{
    MultiComponentGradient(int n) : mN(n)
    {

    }

    //! @brief Gradient for multicomponent dof
    //!
    //! Want to have grad V = B * NodeVals (dofComponents * spaceDim)
    //!
    //! grad V = [ dV1/dx, dV1/dy dV2/dx, dV2/dy ... ]
    //!
    //! With NodeVals = [N1a, N1b, N1c ,N2a , ...] (numNodes x numDof)
    //!
    //! this results in a B-Matrix like this: (mN*dim, mN*numNodes)
    //!
    //! B = |  dN1/dX   0       0  dN2/dX   0      0  dN3/dX   0     0    |
    //!     |  dN1/dY   0       0  dN2/dY   0      0  dN3/dY   0     0    |
    //!     |     0    dN1/dX   0     0    dN2/dX  0     0   dN3/dX  0    |
    //!     |     0    dN1/dY   0     0    dN2/dY  0     0   dN3/dY  0    |
    //!     |     0     0    dN1/dX   0     0    dN2/dX  0     0   dN3/dX |
    //!     |     0     0    dN1/dY   0     0    dN2/dY  0     0   dN3/dY |
    //!
    //! with dNdX (numNodes x spaceDim)
    //!
    Eigen::MatrixXd operator()(const Eigen::MatrixXd& dNdX) const override
    {
        const int dim = dNdX.cols();
        const int numNodes = dNdX.rows();
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mN*dim, mN*numNodes);
        for (int i=0; i<numNodes;i++)
            for (int j=0;j<mN;j++)
                result.block(dim*j,j + mN*i,dim,1) = dNdX.row(i).transpose();
        return result;
    }

    int mN;
};

struct VectorGradientCOOS : Interface
{
    VectorGradientCOOS(CoordinateSystem<3> cs) : mCS(cs)
    {

    }

    //! @brief Gradient for vector dof
    //!
    //! Want to have grad V = B * NodeVals
    //!
    //! With NodeVals = [N1x, N1y, Nz ,N2x , ...] (numNodes x dim)
    //!
    //! Formula is (covariant derivative):
    //!
    //!  v^{i}_{ ;j}   =    dvi / duj + v^l Gamma^i_{jl}
    //!
    //! with dNdX (numNodes x spaceDim)
    //!
    Eigen::MatrixXd operator()(const Eigen::MatrixXd& dNdX) const override
    {
        const int dim = dNdX.cols();
        const int numNodes = dNdX.rows();
        Eigen::MatrixXd B = MultiComponentGradient(dim)(dNdX);

        Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(dim*dim,dim);
        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                for (int k=0;k<3;k++)
                    Gamma(dim*i+j,k) = mCS.GetChristoffelSymbols()[i](j,k);
        assert(B.rows() == Gamma.rows());
        assert(B.cols() == Gamma.cols() * numNodes);
        for (int i=0;i<3;i++)
            B.block(0,i*dim, dim*dim,dim) += Gamma;
        return B;
    }

    CoordinateSystem<3>& mCS;
};

} /* B */
} /* NuTo */
