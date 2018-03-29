#pragma once

#include "nuto/mechanics/mesh/MeshGmsh.h"
#include "CoordinateSystem.h"
#include <iostream>

template<int TDim>
class OrthonormalCoordinateSystem : public CoordinateSystem<TDim> {

public:
  //! @brief Orthonormalized Coordinate system
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  OrthonormalCoordinateSystem(Eigen::Matrix3d J) : CoordinateSystem<TDim>(J) {}

  virtual Eigen::Matrix3d GetMetric() override
  {
       Eigen::Matrix3d metric = Eigen::Matrix3d::Zero();
       for (int i=0; i<3;i++)
       {
           metric(i,i) = OrthonormalCoordinateSystem<TDim>::mJ.col(i).norm();
       }
       return metric;
  }

};
