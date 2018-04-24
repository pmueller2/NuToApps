#pragma once

#include <Eigen/Dense>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_resize.hpp>

namespace Eigen {

template <typename D1, typename D2>
inline const typename Eigen::CwiseBinaryOp<
    typename Eigen::internal::scalar_quotient_op<
        typename Eigen::internal::traits<D1>::Scalar>,
    const D1, const D2>
operator/(const Eigen::MatrixBase<D1> &x1, const Eigen::MatrixBase<D2> &x2) {
  return x1.cwiseQuotient(x2);
}

template <typename D>
inline const typename Eigen::CwiseUnaryOp<
    typename Eigen::internal::scalar_abs_op<
        typename Eigen::internal::traits<D>::Scalar>,
    const D>
abs(const Eigen::MatrixBase<D> &m) {
  return m.cwiseAbs();
}

VectorXd operator+(VectorXd m, double d) {
  return m + VectorXd::Constant(m.rows(), d);
}

VectorXd operator+(double d, VectorXd m) {
  return m + VectorXd::Constant(m.rows(), d);
}

} // end Eigen namespace

typedef Eigen::VectorXd state_type;

namespace boost {
namespace numeric {
namespace odeint {

// needed for steppers with error control
template <> struct vector_space_norm_inf<state_type> {
  typedef double result_type;
  double operator()(const state_type &m) const {
    return m.template lpNorm<Eigen::Infinity>();
  }
};
}
}
}
