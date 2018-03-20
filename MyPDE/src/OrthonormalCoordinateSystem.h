#pragma once

#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "CoordinateSystem.h"
#include <iostream>

class OrthonormalCoordinateSystem : public CoordinateSystem {

public:
  //! @brief Orthonormalized Coordinate system
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  OrthonormalCoordinateSystem(Eigen::Matrix3d J) : CoordinateSystem(J) {}

  virtual Eigen::Matrix3d GetMetric() override
  {
       Eigen::Matrix3d metric = Eigen::Matrix3d::Zero();
       for (int i=0; i<3;i++)
       {
           metric(i,i) = mJ.col(i).norm();
       }
       return metric;
  }

};
