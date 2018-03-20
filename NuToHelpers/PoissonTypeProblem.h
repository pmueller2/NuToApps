#pragma once

#include "nuto/mechanics/dofs/DofType.h"
#include "nuto/mechanics/dofs/DofVector.h"
#include "nuto/mechanics/dofs/DofMatrix.h"
#include "nuto/mechanics/interpolation/TypeDefs.h"
#include "nuto/mechanics/cell/CellIpData.h"

#include <functional>

namespace NuTo
{
namespace Integrands
{

template<int TDim>
class PoissonTypeProblem
{
public:
    PoissonTypeProblem(DofType dof)
        : mDof(dof)
    {
    }

    DofMatrix<double> MassMatrix(const CellIpData& cipd)
    {
        Eigen::MatrixXd N = cipd.N(mDof);
        DofMatrix<double> massLocal;
        massLocal(mDof, mDof) = N.transpose() * N;
        return massLocal;
    }

    DofMatrix<double> StiffnessMatrix(const CellIpData& cipd)
    {
        Eigen::MatrixXd B = cipd.B(mDof,Nabla::Gradient());
        DofMatrix<double> stiffnessLocal;
        stiffnessLocal(mDof, mDof) = B.transpose() * B;
        return stiffnessLocal;
    }

    DofVector<double> LoadVector(const CellIpData& cipd)
    {
        Eigen::MatrixXd N = cipd.N(mDof);
        DofVector<double> loadLocal;
        loadLocal[mDof] = N.transpose() * 0.;
        return loadLocal;
    }

    DofVector<double> LoadVector(const CellIpData& cipd, std::function<double(Eigen::Matrix<double,TDim,1>)> f)
    {
        Eigen::MatrixXd N = cipd.N(mDof);
        DofVector<double> loadLocal;
        loadLocal[mDof] = N.transpose() * f(cipd.GlobalCoordinates());
        return loadLocal;
    }

    DofVector<double> NormalNeumannLoad(const CellIpData& cipd, std::function<Eigen::Matrix<double,TDim,1>(Eigen::Matrix<double,TDim,1>)> f)
    {
        Eigen::MatrixXd N = cipd.N(mDof);
        DofVector<double> loadLocal;

        double normalComponent = f(cipd.GlobalCoordinates()).dot(cipd.GetJacobian().Normal());

        loadLocal[mDof] = N.transpose() * normalComponent;
        return loadLocal;
    }

private:
    DofType mDof;
};

} /* Integrand */
} /* NuTo */

