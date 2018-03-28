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

  //! @brief Coordinate system, zero curvature (second derivatives zero)
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  CoordinateSystem(Eigen::Matrix3d J) : mJ(J) {
    for (int i = 0; i < 3; i++) {
      mK.push_back(Eigen::Matrix3d::Zero());
    }
  }

  virtual Eigen::Matrix3d GetMetric() { return mJ.transpose() * mJ; }

  virtual Eigen::Matrix3d GetJ() { return mJ; }

  //! @brief ChristoffelSymbols 2nd kind
  //!
  //! The symbol \Gamma^i_{jk} is the coefficient i
  //! of the derivative of a tangent vector (vector j in direction k)
  //! The symbols are (usually) symmetric in the two lower indices
  //!
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
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

  //------------------------ Vector transforms -------------------------

  //! @brief Transforms vector from cartesian to tangent basis
  Eigen::Vector3d TransformVectorCtoT(Eigen::VectorXd v) {
    return mJ.inverse() * v;
  }

  //! @brief Transforms vector from cartesian to gradient basis
  Eigen::Vector3d TransformVectorCtoG(Eigen::VectorXd v) {
    return mJ.transpose() * v;
  }

  //! @brief Transforms vector from tangent to cartesian basis
  Eigen::Vector3d TransformVectorTtoC(Eigen::VectorXd v) { return mJ * v; }

  //! @brief Transforms vector from gradient to cartesian basis
  Eigen::Vector3d TransformVectorGtoC(Eigen::VectorXd v) {
    return (mJ.inverse()).transpose() * v;
  }

  //! @brief Transforms vector from gradient to tangent basis
  Eigen::Vector3d TransformVectorGtoT(Eigen::VectorXd v) {
    return GetMetric().inverse() * v;
  }

  //! @brief Transforms vector from tangent to gradient basis
  Eigen::Vector3d TransformVectorTtoG(Eigen::VectorXd v) {
    return GetMetric() * v;
  }

  //------------------------ Tensor transforms -------------------------
  //! @brief Transforms vector from cartesian to tangent basis
  Eigen::Matrix3d TransformTensorCCtoTT(Eigen::Matrix3d v) {
    return mJ.inverse() * v * mJ.inverse().transpose();
  }

  //! @brief Transforms vector from cartesian to gradient basis
  Eigen::Matrix3d TransformTensorCCtoGG(Eigen::Matrix3d v) {
      return mJ.transpose() * v * mJ;
  }

  //! @brief Transforms vector from tangent to cartesian basis
  Eigen::Matrix3d TransformTensorTTtoCC(Eigen::Matrix3d v) {
      return mJ * v * mJ.transpose();
  }

  //! @brief Transforms vector from gradient to cartesian basis
  Eigen::Matrix3d TransformTensorGGtoCC(Eigen::Matrix3d v) {
      return (mJ.inverse()).transpose() * v * mJ.inverse();
  }

  //! @brief Transforms vector from gradient to tangent basis
  Eigen::Matrix3d TransformTensorGGtoTT(Eigen::Matrix3d v) {
      return GetMetric().inverse() * v * GetMetric().inverse().transpose();
  }

  //! @brief Transforms vector from tangent to gradient basis
  Eigen::Matrix3d TransformTensorTTtoGG(Eigen::Matrix3d v) {
      return GetMetric() * v * GetMetric().transpose();
  }

protected:
  Eigen::Matrix3d mJ;

private:
  std::vector<Eigen::Matrix3d> mK;
};
