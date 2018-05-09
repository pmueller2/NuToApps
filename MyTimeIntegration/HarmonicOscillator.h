#include <eigen3/Eigen/Core>

//! @brief Harmonic oscillator with periodic forcing and damping
//! Equation is:
//!  mu'' + ku + beta u' = F cos ωt
//!
//! Written as explicit second order system:
//!   u''  = (F cos ωt - ku - beta u')/m
//!
//! Written as explicit first order system:
//!   u' = v
//!   v' = (F cos ωt - ku - beta v)/m
//!
class HarmonicOscillator {

  double mM;
  double mK;
  double mBeta;
  double mOmega;
  double mF;

  // Derived quantities
  double mEigenFreq;
  double mGamma;
  double mOmegaMod;

public:
  HarmonicOscillator(double m, double k, double beta, double omega, double F)
      : mM(m), mK(k), mBeta(beta), mOmega(omega), mF(F) {
    mEigenFreq = sqrt(mK / mM);
    mGamma = mBeta * mEigenFreq / (2 * mK);
    mOmegaMod = mOmega / mEigenFreq;
  }

  Eigen::Vector2d FirstOrderSystem(Eigen::Vector2d vals, double t) {
    double u = vals[0];
    double v = vals[1];

    double dudt = v;
    double dvdt = (mF * cos(mOmega * t) - mK * u - mBeta * v) / mM;
    return Eigen::Vector2d(dudt, dvdt);
  }

  double SecondOrderSystem(double u, double v, double t) {
    double d2udt2 = (mF * cos(mOmega * t) - mK * u - mBeta * v) / mM;
    return d2udt2;
  }

  double ExactSolution(double u0, double v0, double t) {
    double omega1 = sqrt(1 - mGamma * mGamma);
    double tau = t;

    double A = 1. / (sqrt((1 - mOmega * mOmega) * (1 - mOmega * mOmega) +
                          4 * mGamma * mGamma * mOmega * mOmega));
    double phi = atan(2. * mGamma * mOmega) / (mOmega * mOmega - 1.);

    double C1 = u0 - A * exp(phi);
    double C2 = (v0 + mGamma * C1) / omega1;
    double xTransient =
        exp(-mGamma * tau) * (C1 * cos(omega1 * tau) + C2 * sin(omega1 * tau));
    double xSteady = A * exp(phi) * cos(mOmega * tau);
    return xTransient + xSteady;
  }
};
