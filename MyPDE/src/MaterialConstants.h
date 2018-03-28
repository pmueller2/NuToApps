#include <iostream>
#include <Eigen/Dense>

namespace NCE51
{
static double density = 7850;

static Eigen::Matrix3d permittivity_T = (Eigen::Matrix3d() << 1.72e-8, 0, 0, //
                                         0, 1.72e-8, 0, //
                                         0, 0, 1.68e-8)
                                                .finished();

static Eigen::Matrix3d permittivity_S = (Eigen::Matrix3d() << 0.802e-8, 0, 0, //
                                         0, 0.802e-8, 0, //
                                         0, 0, 0.729e-8)
                                                .finished();

static Eigen::Matrix<double, 6, 6> stiffness_E =
        (Eigen::Matrix<double, 6, 6>() << 13.4 * 1e10, 8.89 * 1e10, 9.09 * 1e10, 0, 0, 0, //
         8.89 * 1e10, 13.4 * 1e10, 9.09 * 1e10, 0, 0, 0, //
         9.09 * 1e10, 9.09 * 1e10, 12.1 * 1e10, 0, 0, 0, //
         0, 0, 0, 2.05 * 1e10, 0, 0, //
         0, 0, 0, 0, 2.05 * 1e10, 0, //
         0, 0, 0, 0, 0, 2.24 * 1e10)
                .finished();

static Eigen::Matrix<double, 6, 6> stiffness_D =
        (Eigen::Matrix<double, 6, 6>() << 13.2 * 1e10, 8.76 * 1e10, 7.34 * 1e10, 0, 0, 0, //
         8.76 * 1e10, 13.2 * 1e10, 7.34 * 1e10, 0, 0, 0, //
         7.34 * 1e10, 7.34 * 1e10, 16.2 * 1e10, 0, 0, 0, //
         0, 0, 0, 4.37 * 1e10, 0, 0, //
         0, 0, 0, 0, 4.37 * 1e10, 0, //
         0, 0, 0, 0, 0, 2.24 * 1e10)
                .finished();

static Eigen::Matrix<double, 3, 6> piezo_d = (Eigen::Matrix<double, 3, 6>() << 0, 0, 0, 0, 669 * 1e-12, 0, //
                                              0, 0, 0, 669 * 1e-12, 0, 0, //
                                              -208 * 1e-12, -208 * 1e-12, 443 * 1e-12, 0, 0,
                                              0).finished();

static Eigen::Matrix<double, 3, 6> piezo_e = (Eigen::Matrix<double, 3, 6>() << 0, 0, 0, 0, 13.7, 0, //
                                              0, 0, 0, 13.7, 0, 0, //
                                              -6.06, -6.06, 17.2, 0, 0,
                                              0).finished();
}

namespace Fuji_C64
{
static double density = 7.7e3; // kg/m^3

static Eigen::Matrix3d relative_permittivity_T = (Eigen::Matrix3d() << 1960, 0, 0, //
                                                  0, 1960, 0, //
                                                  0, 0,
                                                  1850).finished();

// SI units [N/m^2], YoungsModulus_Eii = 1/complianceii
static double YoungsModulus_E11(5.9e10);
static double YoungsModulus_E33(5.1e10);
static double YoungsModulus_E55(2.0e10);

static double PoissonsRation(0.34);

static double piezo_d31(-185);
static double piezo_d33(435);
static double piezo_d15(670);
}
