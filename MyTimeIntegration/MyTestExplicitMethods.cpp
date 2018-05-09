#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

#include "boost/numeric/odeint.hpp"
#include "nuto/math/EigenOdeintCompatibility.h"

#include "../MyTimeIntegration/NY4NoVelocity.h"
#include "../MyTimeIntegration/RK4.h"
#include "../MyTimeIntegration/RK4Nystroem.h"

typedef double state_type;

struct push_back_state_and_time {
  std::vector<state_type> &m_states;
  std::vector<double> &m_times;

  push_back_state_and_time(std::vector<state_type> &states,
                           std::vector<double> &times)
      : m_states(states), m_times(times) {}

  void operator()(const state_type &x, double t) {
    m_states.push_back(x);
    m_times.push_back(t);
  }
};

int main(int /* argc */, char ** /* argv */) {

  double t0 = 0.;
  double tF = 10.;
  double n = 1000;

  double h = (tF - t0) / n;

  // params
  double a = 0.1;
  double k = 4.;
  double gam = 0.2;

  // initial data
  double u0 = 1.2;

  using namespace boost::numeric::odeint;

  typedef runge_kutta_cash_karp54<state_type, double, state_type, double,
                                  vector_space_algebra>
      stepper_type;
  stepper_type stepper;

  std::vector<state_type> x_vec;
  std::vector<double> times;

  // dx/dt = -a x t
  //
  // result is x0 exp(-a t*t/2)
  //
  auto eq1 = [&](const state_type &x, state_type &dxdt, const double t) {
    dxdt = -a * x * t;
  };

  auto result1 = [&](double t) { return u0 * exp(-a * t * t / 2.); };

  // d2x/dt2 = -k x
  //
  // result is cos(omega t), omega = sqrt(k)
  //
  auto eq2 = [&](const state_type &x, state_type &d2xdt2, const double t) {
    d2xdt2 = -k * x;
  };

  auto result2 = [&](double t) { return u0 * cos(sqrt(k) * t); };

  // d2x/dt2 = -k x - gam x'
  //
  // result is cos(omega t), omega = sqrt(k)
  //
  auto eq3 = [&](const state_type &x, const state_type &v, state_type &d2xdt2,
                 const double t) { d2xdt2 = -x - 2. * gam * v; };

  auto result3 = [&](double t) {
    double om = sqrt(1 - gam * gam);
    return u0 * exp(-gam * t) * (cos(om * t) + gam * sin(om * t));
  };

  state_type x0 = u0;

  // ************** ODEINT STEPPER **************************

  //  integrate_adaptive(stepper, eq1, x0, t0, tF, h,
  //                     push_back_state_and_time(x_vec, times));

  //  for (size_t i = 0; i < times.size(); i++) {
  //    double t = times[i];
  //    std::cout << t << '\t' << x_vec[i] << '\t' << result1(t) << '\n';
  //  }

  // ************** OWN FIRST ORDER SYSTEM SOLVER*************

  //  NuTo::TimeIntegration::RK4<double> ti;

  //  for (int i = 0; i < n; i++) {
  //    double t = t0 + i * h;
  //    ti.do_step(eq1, x0, t, h);
  //    std::cout << t << '\t' << x0 << '\t' << result1(t) << '\n';
  //  }

  // ************** OWN SECOND ORDER NO VELOCITY SOLVER*************

  //  NuTo::TimeIntegration::NY4NoVelocity<double> ti;

  //  auto xv0 = std::make_pair(u0, 0.);

  //  for (int i = 0; i < n; i++) {
  //    double t = t0 + i * h;
  //    xv0 = ti.DoStep(eq2, xv0.first, xv0.second, t, h);
  //    std::cout << t << '\t' << xv0.first << '\t' << result2(t) << '\n';
  //  }

  // ************** OWN SECOND ORDER SOLVER*************

  NuTo::TimeIntegration::RK4Nystroem<double> ti;

  auto xv0 = std::make_pair(u0, 0.);

  for (int i = 0; i < n; i++) {
    double t = t0 + i * h;
    xv0 = ti.DoStep(eq3, xv0.first, xv0.second, t, h);
    std::cout << t << '\t' << xv0.first << '\t' << result3(t) << '\n';
  }
}
