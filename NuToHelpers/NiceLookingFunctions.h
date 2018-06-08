#pragma once

double cosineBump(double x) {
  double a = 0.3;
  double b = 0.7;
  if ((a < x) && (x < b)) {
    return 0.5 * (1. - cos(2 * M_PI * (x - a) / (b - a)));
  }
  return 0.;
}

double cosineBumpDerivative(double x) {
  double a = 0.3;
  double b = 0.7;
  if ((a < x) && (x < b)) {
    return M_PI / (b - a) * sin(2 * M_PI * (x - a) / (b - a));
  }
  return 0.;
}

double smearedStepFunction(double t, double tau) {
  if ((0 < t) && (t < tau)) {
    return 0.5 * (1. - cos(2 * M_PI * t / tau));
  }
  return 0.;
}
