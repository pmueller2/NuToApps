#pragma once

#include "CoordinateSystem.h"
#include "OrthonormalCoordinateSystem.h"
#include <iostream>

using namespace NuTo;

//! @brief Spherical coordinates with the following conventions
//!
//! Coordinates (u,v,w) = (r, theta, phi), ...???
//!
class SphericalCoordinates {
public:
  Eigen::Vector3d GetCartesian(Eigen::Vector3d coords) {
    double r = coords[0];
    double theta = coords[1];
    double phi = coords[2];

    double x = r * sin(theta) * cos(phi);
    double y = r * sin(theta) * sin(phi);
    double z = r * cos(theta);

    return Eigen::Vector3d(x, y, z);
  }

  CoordinateSystem<3> GetNaturalCOOS(Eigen::Vector3d coords) {
    // unpack
    double r = coords[0];
    double theta = coords[1];
    double phi = coords[2];

    // Jacobian
    Eigen::Matrix3d J;
    J << sin(theta) * cos(phi), r * cos(phi) * cos(theta),
        -r * sin(phi) * sin(theta), //
        sin(phi) * sin(theta), r * sin(phi) * cos(theta),
        r * sin(theta) * cos(phi),      //
        cos(theta), -r * sin(theta), 0; //

    // 2nd Derivatives
    Eigen::Matrix3d K1;
    K1 << 0, cos(phi) * cos(theta), -sin(phi) * sin(theta), //
        cos(phi) * cos(theta), -r * sin(theta) * cos(phi),
        -r * sin(phi) * cos(theta), //
        -sin(phi) * sin(theta), -r * sin(phi) * cos(theta),
        -r * sin(theta) * cos(phi);

    Eigen::Matrix3d K2;
    K2 << 0, sin(phi) * cos(theta), sin(theta) * cos(phi), //
        sin(phi) * cos(theta), -r * sin(phi) * sin(theta),
        r * cos(phi) * cos(theta), //
        sin(theta) * cos(phi), r * cos(phi) * cos(theta),
        -r * sin(phi) * sin(theta);

    Eigen::Matrix3d K3;
    K3 << 0, -sin(theta), 0,             //
        -sin(theta), -r * cos(theta), 0, //
        0, 0, 0;

    return CoordinateSystem<3>(J, {K1, K2, K3});
  }

  OrthonormalCoordinateSystem<3> GetOrthonormalCOOS(Eigen::Vector3d coords) {}
};
