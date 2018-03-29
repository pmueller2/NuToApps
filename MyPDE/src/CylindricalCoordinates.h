#pragma once

#include "CoordinateSystem.h"
#include "OrthonormalCoordinateSystem.h"
#include <iostream>

using namespace NuTo;

//! @brief Cylindrical coordinates with the following conventions
//!
//! Coordinates (u,v,w) = (r, phi , zeta), r<0, phi ?,
//!
class CylindricalCoordinates {
public:
  Eigen::Vector3d GetCartesian(Eigen::Vector3d coords) {
    double r = coords[0];
    double phi = coords[1];
    double zeta = coords[2];

    double x = r * cos(phi);
    double y = r * sin(phi);
    double z = zeta;
    return Eigen::Vector3d(x, y, z);
  }

  CoordinateSystem<3> GetNaturalCOOS(Eigen::Vector3d coords) {
    // unpack
    double r = coords[0];
    double phi = coords[1];
    double zeta = coords[2];

    // Jacobian
    Eigen::Matrix3d J;
    J << cos(phi), -r * sin(phi), 0, //
        sin(phi), r * cos(phi), 0,   //
        0, 0, 1;

    // 2nd Derivatives
    Eigen::Matrix3d K1;
    K1 << 0, -sin(phi), 0,           //
        -sin(phi), -r * cos(phi), 0, //
        0, 0, 0;                     //

    Eigen::Matrix3d K2;
    K2 << 0, cos(phi), 0,           //
        cos(phi), -r * sin(phi), 0, //
        0, 0, 0;                    //

    Eigen::Matrix3d K3;
    K3.setZero();

    return CoordinateSystem<3>(J, {K1, K2, K3});
  }

  OrthonormalCoordinateSystem<3> GetOrthonormalCOOS(Eigen::Vector3d coords) {}
};
