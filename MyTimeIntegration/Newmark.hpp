#pragma once

#include "math/NewtonRaphson.h"
#include <eigen3/Eigen/Dense>

template <typename Tstate, typename Tmxstate> class Newmark {

public:
  //! @brief Performs one Newmark step
  //!
  //! For the second order differential equation f(y'',y',y,t) = 0
  //!
  //! @param f A functor that returns the right hand side of the differential
  //! equation
  //!
  //! The signature of its call operator must be:
  //! operator()(const Tstate& w, const Tstate& v, Tstate& res, double t)
  //! The return value (residuum) is stored in res
  //!
  //! @param J0, J1, J2 A functor that returns the Jacobians of the right hand side of
  //! the differential equation (J0 = df/dy, J1 = df/dy', J2 = df/dy'')
  //!
  //! The signature of its call operator must be:
  //! operator()(const Tstate& w, const Tstate& v, Tmxstate& J, double t)
  //! The return value is stored in dwdt
  //!
  //! @param w0 initial value
  //! @param v0 initial velocity
  //! @param J return argument
  //! @param t0 start time
  //! @param h step size (t-t0)
  //! @return value after one Euler step
  template <typename F, typename DF>
  Tstate DoStep(F f, DF J0, DF J1,  DF J2, Tstate w0, Tstate v0, Tstate a0, Tstate initialGuess, double t0,
                double h) {

    double t = t0 + h;

    auto a = a0;
    auto v = v0;
    auto x = initialGuess;

    auto ResidualFunction = [&](Tstate xNew){
        auto aNew = 1./(beta*h)*( (xNew-x)/h - v) + (1.-1./(2.*beta))*a;
        auto vNew = v + h*(1.-gamma/(2.*beta))*a + gamma/beta*( (xNew-x)/h - v);
        Tstate res;
        f(aNew,vNew,x,res,t);
        return (res);
    };

    auto DerivativeFunction = [&](Tstate xNew){
        auto aNew = 1./(beta*h)*( (xNew-x)/h - v) + (1.-1./(2.*beta))*a;
        auto vNew = v + h*(1.-gamma/(2.*beta))*a + gamma/beta*( (xNew-x)/h - v);
        Tmxstate j0;
        Tmxstate j1;
        Tmxstate j2;
        J0(aNew,vNew,x,j0,t);
        J1(aNew,vNew,x,j1,t);
        J2(aNew,vNew,x,j2,t);
        Tmxstate result = j2/(beta*h*h) + j1*gamma/(beta*h) + j0;
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

    auto result = NuTo::NewtonRaphson::Solve(problem,initialGuess,EigenSolverWrapper());

    return result;
  }

  double beta = 0.25;
  double gamma = 0.5 ;
};

