#include <iostream>
#include <vector>

#include "BDF.hpp"
#include "ImplicitEuler.hpp"
#include "Newmark.hpp"
#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<double, 2, 1> Tstate;
typedef Eigen::Matrix<double, 2, 2> Tmxstate;

class HarmonicOscillator {
public:
  HarmonicOscillator(double freq) : k(4. * M_PI * M_PI * freq * freq) {}

  void operator()(const Tstate &w, Tstate &dwdt, double t) {
    dwdt[0] = w[1];
    dwdt[1] = -k * w[0];
  }

  Tstate exact(double t) {
    Tstate result;
    result[0] = cos(sqrt(k) * t);
    result[1] = -sqrt(k) * sin(sqrt(k) * t);
    return result;
  }

  double k;
};

class HarmonicOscillatorJacobian {
public:
  HarmonicOscillatorJacobian(double k) : k(k) {}

  void operator()(const Tstate &w, Tmxstate &dwdt, double t) {
    dwdt.setZero();
    dwdt(0, 1) = 1.;
    dwdt(1, 0) = -k;
  }

  double k;
};

int main(int /* argc */, char ** /* argv */) {
  HarmonicOscillator eq(5.);
  HarmonicOscillatorJacobian jac(5.);

  int n = 1000;
  double h = 0.001;

  // Exact result
  for (int i = 0; i < n; i++) {
    double result = eq.exact(i * h)[0];
    std::cout << result << std::endl;
  }

  // Euler implicit result
  std::cout << std::endl;
  std::cout << std::endl;

  ImplicitEuler<Tstate, Tmxstate> ti2;

  Eigen::VectorXd w(2);
  w << 1., 0.;
  Tstate guess;
  guess = w;

  for (int i = 0; i < n; i++) {
    Tstate result = ti2.DoStep(eq, jac, w, guess, i * h, h);
    w = result;
    guess = result;
    std::cout << result[0] << std::endl;
  }

  // BDF result
  std::cout << std::endl;
  std::cout << std::endl;

  int nBDF = 4;
  BDF<Tstate, Tmxstate> ti3(nBDF);

  // Get initial states
  std::vector<Tstate> w0Vector;
  for (int i = 0; i < nBDF; i++) {
    w0Vector.push_back(eq.exact(i * h));
  }
  guess = w0Vector[nBDF - 1];

  // plot the first initial steps
  for (int i = 0; i < nBDF; i++) {
    std::cout << w0Vector[i][0] << std::endl;
  }

  for (int i = nBDF; i < n; i++) {
    Tstate result = ti3.DoStep(eq, jac, w0Vector, guess, i * h, h);
    // Add result and remove oldest
    w0Vector.push_back(result);
    w0Vector.erase(w0Vector.begin());
    guess = result;
    std::cout << result[0] << std::endl;
  }
}
