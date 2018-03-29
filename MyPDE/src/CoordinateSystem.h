#pragma once

#include "nuto/mechanics/mesh/MeshGmsh.h"
#include <iostream>

using namespace NuTo;

template<int TDim>
class CoordinateSystem {

    using Matrix = Eigen::Matrix<double,TDim,TDim>;
    using Vector = Eigen::Matrix<double,TDim,1>;

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
  CoordinateSystem(Matrix J, std::vector<Matrix> K) : mJ(J) {
    for (int i = 0; i < TDim; i++) {
      mK.push_back(K[i]);
    }
  }

  //! @brief Coordinate system, zero curvature (second derivatives zero)
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  CoordinateSystem(Matrix J) : mJ(J) {
    for (int i = 0; i < TDim; i++) {
      mK.push_back(Matrix::Zero());
    }
  }

  virtual Matrix GetMetric() { return mJ.transpose() * mJ; }

  virtual Matrix GetJ() { return mJ; }

  virtual double GetDetJ() { return mJ.determinant(); }

  //! @brief ChristoffelSymbols 2nd kind
  //!
  //! The symbol \Gamma^i_{jk} is the coefficient i
  //! of the derivative of a tangent vector (vector j in direction k)
  //! The symbols are (usually) symmetric in the two lower indices
  //!
  //! @param J columns represent basis vectors (tangential basis)
  //! in cartesian coordinates
  virtual std::vector<Matrix> GetChristoffelSymbols() {
    std::vector<Matrix> result(TDim);
    Matrix Jinv = mJ.inverse();
    for (int i = 0; i < TDim; i++) {
      Matrix gamma = Matrix::Zero();
      for (int l = 0; l < TDim; l++)
        gamma += mK[i] * Jinv(i, l);
      result[i] = gamma;
    }
    return result;
  }

  //------------------------ Vector transforms -------------------------

  //! @brief Transforms vector from cartesian to tangent basis
  Vector TransformVectorCtoT(Eigen::VectorXd v) {
    return mJ.inverse() * v;
  }

  //! @brief Transforms vector from cartesian to gradient basis
  Vector TransformVectorCtoG(Eigen::VectorXd v) {
    return mJ.transpose() * v;
  }

  //! @brief Transforms vector from tangent to cartesian basis
  Vector TransformVectorTtoC(Eigen::VectorXd v) { return mJ * v; }

  //! @brief Transforms vector from gradient to cartesian basis
  Vector TransformVectorGtoC(Eigen::VectorXd v) {
    return (mJ.inverse()).transpose() * v;
  }

  //! @brief Transforms vector from gradient to tangent basis
  Vector TransformVectorGtoT(Eigen::VectorXd v) {
    return GetMetric().inverse() * v;
  }

  //! @brief Transforms vector from tangent to gradient basis
  Vector TransformVectorTtoG(Eigen::VectorXd v) {
    return GetMetric() * v;
  }

  //------------------------ Tensor transforms -------------------------
  //! @brief Transforms vector from cartesian to tangent basis
  Matrix TransformTensorCCtoTT(Matrix v) {
    return mJ.inverse() * v * mJ.inverse().transpose();
  }

  //! @brief Transforms vector from cartesian to gradient basis
  Matrix TransformTensorCCtoGG(Matrix v) {
      return mJ.transpose() * v * mJ;
  }

  //! @brief Transforms vector from tangent to cartesian basis
  Matrix TransformTensorTTtoCC(Matrix v) {
      return mJ * v * mJ.transpose();
  }

  //! @brief Transforms vector from gradient to cartesian basis
  Matrix TransformTensorGGtoCC(Matrix v) {
      return (mJ.inverse()).transpose() * v * mJ.inverse();
  }

  //! @brief Transforms vector from gradient to tangent basis
  Matrix TransformTensorGGtoTT(Matrix v) {
      return GetMetric().inverse() * v * GetMetric().inverse().transpose();
  }

  //! @brief Transforms vector from tangent to gradient basis
  Matrix TransformTensorTTtoGG(Matrix v) {
      return GetMetric() * v * GetMetric().transpose();
  }

protected:
  Matrix mJ;

private:
  std::vector<Matrix> mK;
};
