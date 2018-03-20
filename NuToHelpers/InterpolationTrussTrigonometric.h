#pragma once
#include <eigen3/Eigen/Core>
#include <cmath>
#include "nuto/mechanics/interpolation/InterpolationSimple.h"
#include "nuto/math/shapes/Line.h"

namespace NuTo
{
class InterpolationTrussTrigonometric : public InterpolationSimple
{
public:
    InterpolationTrussTrigonometric(int numNodes) : mNumNodes(numNodes)
    {

    }

    std::unique_ptr<InterpolationSimple> Clone() const override
    {
        return std::make_unique<InterpolationTrussTrigonometric>(*this);
    }

    ShapeFunctions GetShapeFunctions(const NaturalCoords& naturalIpCoords) const override
    {
        int N = mNumNodes -1;
        Eigen::VectorXd shapes(mNumNodes-1);
        //double x = naturalIpCoords[0];
        double x = (naturalIpCoords[0] + 1.)/2 * 2*M_PI;
        for (int i=0; i<shapes.size();i++)
        {
            double xi = i*2.*M_PI/N;
            if (std::abs(xi-x) < 1e-8)
            {
                shapes.setZero();
                shapes[i] = 1.;
                //shapes[N] = shapes[0];
                //shapes[N] = 0.;
                return shapes;
            }
            shapes[i] = sin(N/2.*(x-xi))/(tan(0.5*(x-xi))) /N;
        }
        //shapes[N] = shapes[0];
        //shapes[N] = 0.;
        return shapes;
    }

    DerivativeShapeFunctionsNatural GetDerivativeShapeFunctions(const NaturalCoords& naturalIpCoords) const override
    {
        Eigen::VectorXd derivativeShapes(mNumNodes-1);
        int N = mNumNodes -1;
        for (int i=0; i<derivativeShapes.size();i++)
        {
            double xi = i*2.*M_PI/N;
            double x = (naturalIpCoords[0] + 1.)/2 * 2*M_PI - xi;
            if ((-cos(x) + 1.)<1e-10) {
                derivativeShapes[i] = 0.;
            } else {
                derivativeShapes[i] = M_PI/N / (-cos(x) + 1.) * ( -N/4. * sin(N*x/2.-x) + N/4. * sin(N*x/2.+x) - sin(N*x/2.) );
            }
        }
        return derivativeShapes;
    }

    NaturalCoords GetLocalCoords(int nodeId) const override
    {
        return Eigen::VectorXd::Constant(1, -1. + 2. * nodeId/(mNumNodes-1));
    }

    int GetNumNodes() const override
    {
        return (mNumNodes-1);
    }

    const Shape& GetShape() const override
     {
         return mShape;
     }

private:
    int mNumNodes;
    Line mShape;
};
} /* NuTo */
