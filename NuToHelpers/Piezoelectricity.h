#pragma once

#include "nuto/mechanics/dofs/DofMatrix.h"
#include "nuto/mechanics/dofs/DofType.h"
#include "nuto/mechanics/dofs/DofVector.h"

#include "nuto/mechanics/cell/CellIpData.h"

#include <Eigen/Core>

namespace NuTo {
namespace Integrands {

class Piezoelectricity {
public:
  Piezoelectricity(Eigen::MatrixXd stiffness, Eigen::MatrixXd piezo,
                   Eigen::MatrixXd permittivity)
      : mC(stiffness), mE(piezo), mP(permittivity), dofV("Potential", 1),
        dofU("Displacements", 3) {}

  DofVector<double> Gradient(const CellIpData &cipd) {

    Eigen::MatrixXd Belectrical = cipd.B(dofV, Nabla::Gradient());
    Eigen::MatrixXd Bstrain = cipd.B(dofV, Nabla::Strain());

    auto EField = cipd.Apply(dofV, Nabla::Gradient());
    auto Strain = cipd.Apply(dofU, Nabla::Strain());

    DofVector<double> gradient;

    gradient[dofV] = -Belectrical.transpose() * mP * EField +
                     Belectrical.transpose() * mE * Strain;
    gradient[dofU] = Bstrain.transpose() * mC * Strain +
                     Bstrain.transpose() * mE.transpose() * EField;

    return gradient;
  }

  DofMatrix<double> Hessian2(const CellIpData &cipd) {

    Eigen::MatrixXd Belectrical = cipd.B(dofV, Nabla::Gradient());
    Eigen::MatrixXd Bstrain = cipd.B(dofU, Nabla::Strain());

    DofMatrix<double> hessian2;

    hessian2(dofU, dofU) = Bstrain.transpose() * mC * Bstrain;
    hessian2(dofU, dofV) = Bstrain.transpose() * mE.transpose() * Belectrical;
    hessian2(dofV, dofU) = Belectrical.transpose() * mE * Bstrain;
    hessian2(dofV, dofV) = -Belectrical.transpose() * mP * Belectrical;

    return hessian2;
  }

  DofVector<double> NeumannLoadElectrical(const CellIpData &cipd, double Dn) {

    Eigen::MatrixXd N = cipd.N(dofV);
    DofVector<double> load;
    load[dofV] = N.transpose() * Dn;
    return load;
  }

  DofVector<double>
  NeumannLoadElectrical(const CellIpData &cipd,
                        std::function<double(Eigen::VectorXd)> Dn) {

    Eigen::MatrixXd N = cipd.N(dofV);
    DofVector<double> load;
    load[dofV] = N.transpose() * Dn(cipd.GlobalCoordinates());
    return load;
  }

  DofVector<double> NeumannLoadMechanical(const CellIpData &cipd,
                                          Eigen::Vector3d traction) {

    Eigen::MatrixXd N = cipd.N(dofU);
    DofVector<double> load;
    load[dofV] = N.transpose() * traction;
    return load;
  }

  DofVector<double> NeumannLoadMechanical(
      const CellIpData &cipd,
      std::function<Eigen::Vector3d(Eigen::VectorXd)> traction) {

    Eigen::MatrixXd N = cipd.N(dofU);
    DofVector<double> load;
    load[dofV] = N.transpose() * traction(cipd.GlobalCoordinates());
    return load;
  }

  DofType dofV;
  DofType dofU;

private:
  Eigen::MatrixXd mC;
  Eigen::MatrixXd mE;
  Eigen::MatrixXd mP;
};
} /* Integrand */
} /* NuTo */
