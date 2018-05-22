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
  OrthonormalCoordinateSystem(Eigen::Matrix<double,TDim,TDim> J) : CoordinateSystem<TDim>(J) {}

  virtual Eigen::Matrix<double,TDim,TDim> GetMetric() override
  {
       auto metric = Eigen::Matrix<double,TDim,TDim>::Zero();
       for (int i=0; i<TDim;i++)
       {
           metric(i,i) = OrthonormalCoordinateSystem<TDim>::mJ.col(i).norm();
       }
       return metric;
  }

};
