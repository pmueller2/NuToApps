#pragma once

#include <vector>
#include "math/NewtonRaphson.h"
#include <eigen3/Eigen/Dense>

template <typename Tstate, typename Tmxstate> class BDF {

public:

    BDF(int n)
    {
        std::vector<std::vector< double>> coeffs = {{ 1      , -1.},
                                                 {3./2   , -2., 1./2},
                                                 {11./6  , -3., 3./2 , -1./3},
                                                 {25./12 , -4., 3.   , -4./3, 1./4},
                                                 {137./60, -5., 5.   , -10./3, 5./4 , -1./5},
                                                 {147./60, -6., 15./2,-20./3, 15./4, -6./5, 1./6} };
        for (int i=n; i>=0; i--)
        {
            a.push_back(coeffs[n-1][i]);
        }
    }

  //! @brief Performs one BDF step
  //! @param f A functor that returns the right hand side of the differential
  //! equation
  //!
  //! The signature of its call operator must be:
  //! operator()(const Tstate& w, Tstate& dwdt, double t)
  //! The return value is stored in dwdt
  //!
  //! @param J A functor that returns the Jacobian of the right hand side of
  //! the differential equation
  //!
  //! The signature of its call operator must be:
  //! operator()(const Tstate& w, Tmxstate& dwdt, double t)
  //! The return value is stored in dwdt
  //!
  //! @param w0 initial values (multistep formula, n previous values needed)
  //! @param t0 start time
  //! @param h step size (t-t0)
  //! @return value after one step
  template <typename F, typename DF> Tstate DoStep(F f, DF df, std::vector<Tstate> w0, Tstate initialGuess, double t0, double h) {
    // number of previous steps should equal number of coefficients
    if (w0.size() != a.size() -1)
        throw;
    int n = a.size();

    double t = t0 + h;

    auto ResidualFunction = [&](Tstate x){
        Tstate dwdt;
        f(x,dwdt,t);
        Tstate result = h * dwdt;
        for (int i=0; i<n-1; i++)
        {
            result -= a[i]*w0[i];
        }
        result -= a[n-1]*x;
        return (result);
    };

    auto DerivativeFunction = [&](Tstate x){
        Tmxstate jac;
        df(x,jac,t);
        Tmxstate result = h*jac-a[n-1]*jac.Identity();
        return result;
    };

    auto Norm = [](Tstate w) { return w.norm(); };

    auto problem = NuTo::NewtonRaphson::DefineProblem(ResidualFunction,DerivativeFunction,Norm,1.e-6);

    class EigenSolverWrapper
    {
    public:

        Tstate Solve(Tmxstate& DR, const Tstate& R)
        {
            Tstate result = DR.colPivHouseholderQr().solve(R);
            return result;
        }
    };

    Tstate x = NuTo::NewtonRaphson::Solve(problem,initialGuess,EigenSolverWrapper());

    return x;
    }

    std::vector<double> a;
};
