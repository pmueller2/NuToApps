#pragma once
#include <eigen3/Eigen/Core>
#include "nuto/mechanics/interpolation/InterpolationSimple.h"
#include "nuto/math/Legendre.h"
#include "nuto/base/Exception.h"
#include "nuto/math/shapes/Quadrilateral.h"

namespace NuTo
{
class InterpolationQuadHierarchical : public InterpolationSimple
{
public:
    InterpolationQuadHierarchical(int degree) : mDegree(degree)
    {
        if (mDegree<1)
            throw Exception(__PRETTY_FUNCTION__,"Degree must be at least 1.");
    }

    std::unique_ptr<InterpolationSimple> Clone() const override
    {
        return std::make_unique<InterpolationQuadHierarchical>(*this);
    }

    Eigen::VectorXd GetShapeFunctions(const NaturalCoords& coords) const override
    {
        double x = coords[0];
        double y = coords[1];

        Eigen::VectorXd vertexShapes(4);
        vertexShapes[0] = 0.25 * (1. - x) * (1. - y);
        vertexShapes[1] = 0.25 * (1. + x) * (1. - y);
        vertexShapes[2] = 0.25 * (1. + x) * (1. + y);
        vertexShapes[3] = 0.25 * (1. - x) * (1. + y);

        Eigen::VectorXd edgeShapes1(mDegree-1);
        Eigen::VectorXd edgeShapes2(mDegree-1);
        Eigen::VectorXd edgeShapes3(mDegree-1);
        Eigen::VectorXd edgeShapes4(mDegree-1);

        if (mDegree > 1)
        {
            for (int i=2; i<=mDegree; i++)
            {
                double Hx = (Math::Polynomial::Legendre(i,x) - Math::Polynomial::Legendre(i-2,x))/sqrt(2.*(2*i-1));
                double Hy = (Math::Polynomial::Legendre(i,y) - Math::Polynomial::Legendre(i-2,y))/sqrt(2.*(2*i-1));

                edgeShapes1[i-2] = 0.5 * (1. - y) * Hx;
                edgeShapes2[i-2] = 0.5 * (1. + x) * Hy;
                edgeShapes3[i-2] = 0.5 * (1. + y) * Hx;
                edgeShapes4[i-2] = 0.5 * (1. - x) * Hy;
            }
        }

        Eigen::VectorXd interiorShapes( (mDegree-2)*(mDegree-3)/2);

        return vertexShapes;
    }

    Eigen::MatrixXd GetDerivativeShapeFunctions(const NaturalCoords& coords) const override
    {
        Eigen::VectorXd derivativeShapes(mDegree+1);
        return derivativeShapes;
    }

    NaturalCoords GetLocalCoords(int nodeId) const override
    {
        throw Exception(__PRETTY_FUNCTION__,"No Coordinate known for node with Id " + std::to_string(nodeId) + ".");
    }

    int GetNumNodes() const override
    {
        return (mDegree+1);
    }

    const Shape& GetShape() const override
     {
         return mShape;
     }

private:
    int mDegree;
    Quadrilateral mShape;
};
} /* NuTo */
