#pragma once

#include "nuto/mechanics/mesh/MeshGmsh.h"
#include <iostream>

using namespace NuTo;

class CoordinateSystem {
public:
  //! @brief Coordinate system
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  //! @param K1 for a curved coordinates system like
  //! {x(u1,u2,u3),y(u1,u2,u3),z(u1,u2,u3)} the second derivatives
  //! d2x/duiduj
  //! @param K2 for a curved coordinates system like
  //! {x(u1,u2,u3),y(u1,u2,u3),z(u1,u2,u3)} the second derivatives
  //! d2y/duiduj
  //! @param K3 for a curved coordinates system like
  //! {x(u1,u2,u3),y(u1,u2,u3),z(u1,u2,u3)} the second derivatives
  //! d2z/duiduj
  CoordinateSystem(Eigen::Matrix3d J, std::vector<Eigen::Matrix3d> K) : mJ(J) {
    for (int i = 0; i < 3; i++) {
      mK.push_back(K[i]);
    }
  }

  //! @brief Coordinate system, zero curvature (Christoffelsymbols zero)
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  CoordinateSystem(Eigen::Matrix3d J) : mJ(J) {
    for (int i = 0; i < 3; i++) {
      mK.push_back(Eigen::Matrix3d::Zero());
    }
  }

  virtual Eigen::Matrix3d GetMetric() { return mJ.transpose() * mJ; }

  virtual Eigen::Matrix3d GetJ() { return mJ; }

  virtual std::vector<Eigen::Matrix3d> GetChristoffelSymbols() {
    std::vector<Eigen::Matrix3d> result(3);
    Eigen::Matrix3d Jinv = mJ.inverse();
    for (int i = 0; i < 3; i++) {
      Eigen::Matrix3d gamma = Eigen::Matrix3d::Zero();
      for (int l = 0; l < 3; l++)
        gamma += mK[i] * Jinv(i, l);
      result[i] = gamma;
    }
    return result;
  }

protected:
  Eigen::Matrix3d mJ;

private:
  std::vector<Eigen::Matrix3d> mK;
};
