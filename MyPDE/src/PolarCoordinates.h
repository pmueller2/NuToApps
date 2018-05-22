#pragma once

#include "CoordinateSystem.h"
#include "OrthonormalCoordinateSystem.h"
#include <iostream>

using namespace NuTo;

//! @brief Polar coordinates with the following conventions
//!
//! Coordinates (u,v,w) = (r, phi), r<0, phi [0,2 pi]
//!
class PolarCoordinates {
public:
  Eigen::Vector2d GetCartesian(Eigen::Vector2d coords) {
    double r = coords[0];
    double phi = coords[1];

    double x = r * cos(phi);
    double y = r * sin(phi);
    return Eigen::Vector2d(x, y);
  }

  CoordinateSystem<2> GetNaturalCOOS(Eigen::Vector2d coords) {
    // unpack
    double r = coords[0];
    double phi = coords[1];

    // Jacobian
    Eigen::Matrix2d J;
    J << cos(phi), -r * sin(phi), //
        sin(phi), r * cos(phi);   //

    // 2nd Derivatives
    Eigen::Matrix2d K1;
    K1 << 0, -sin(phi),           //
        -sin(phi), -r * cos(phi); //

    Eigen::Matrix2d K2;
    K2 << 0, cos(phi),           //
        cos(phi), -r * sin(phi); //

    return CoordinateSystem<2>(J, {K1, K2});
  }

  OrthonormalCoordinateSystem<2> GetOrthonormalCOOS(Eigen::Vector2d coords) {}
};
