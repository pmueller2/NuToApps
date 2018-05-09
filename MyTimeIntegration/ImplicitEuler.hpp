#pragma once

#include "nuto/math/NewtonRaphson.h"
#include <eigen3/Eigen/Dense>

template <typename Tstate, typename Tmxstate> class ImplicitEuler {

public:
  //! @brief Performs one implicit Euler step
  //!
  //! For the first order differential equation y' = f(y,t)
  //! an implicit Euler step is:
  //! y1 = y0 + f(y1,t1)
  //!
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
  //! @param w0 initial value
  //! @param t0 start time
  //! @param h step size (t-t0)
  //! @return value after one Euler step
  template <typename F, typename DF>
  Tstate DoStep(F f, DF df, Tstate w0, Tstate initialGuess, double t0,
                double h) {

    double t = t0 + h;

    auto ResidualFunction = [&](Tstate w) {
      Tstate dwdt;
      f(w, dwdt, t);
      Tstate result = w0 - w + h * dwdt;
      return (result);
    };

    auto DerivativeFunction = [&](Tstate w) {
      Tmxstate jac;
      df(w, jac, t);
      Tmxstate result = h * jac - jac.Identity(w.size(), w.size());
      return result;
    };

    auto Norm = [](Tstate w) { return w.norm(); };

    auto problem = NuTo::NewtonRaphson::DefineProblem(
        ResidualFunction, DerivativeFunction, Norm, 1.e-6);

    class EigenSolverWrapper {
    public:
      Tstate Solve(Tmxstate &DR, const Tstate &R) {
        Tstate result = DR.colPivHouseholderQr().solve(R);
        return result;
      }
    };

    auto result =
        NuTo::NewtonRaphson::Solve(problem, initialGuess, EigenSolverWrapper());

    return result;
  }
};
