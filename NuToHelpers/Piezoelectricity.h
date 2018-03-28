#pragma once

#include "nuto/mechanics/dofs/DofMatrix.h"
#include "nuto/mechanics/dofs/DofType.h"
#include "nuto/mechanics/dofs/DofVector.h"

#include "nuto/mechanics/cell/CellIpData.h"

#include <Eigen/Core>

namespace NuTo {
namespace Integrands {

/* Integrand for piezoelectricity computations
 *
 * Piezoelectric constitutive law: Parameters assume stress-strain form:
 *
 * Stress = C * strain - E^T * EField
 * DField = P * EField + E   * strain
 *
 * with stiffness matrix C (at constant E),
 * piezo tensor E, permittivity P (at constant strain)
 *
 * Dofs:
 *
 * Potential     EField =  - gradient dofV
 * Displacements strain = symGradient dofU
 *
 * Governing equations:
 *
 * Electrostatics                          div DField = charge
 * Mechanical dynamics  density * d2u/dt2             = div stress + fext
 *
 * Weak form:
 *
 * Electrostatics                                 - (grad N) DField = psi charge - int_Gamma (psi * DField * normal)
 * Mechanical dynamics       density * N d2u/dt2  + (grad N) stress =            + int_Gamma stress * normal + psi fext
*/
class Piezoelectricity {
public:
  Piezoelectricity(Eigen::MatrixXd stiffness, Eigen::MatrixXd piezo,
                   Eigen::MatrixXd permittivity)
      : mC(stiffness), mE(piezo), mP(permittivity), dofV("Potential", 1),
        dofU("Displacements", 3) {}

  Eigen::VectorXd EField(const CellIpData &cipd)
  {
      Eigen::VectorXd EField =-cipd.Apply(dofV, Nabla::Gradient());
      return EField;
  }

  Eigen::VectorXd Strain(const CellIpData &cipd)
  {
      Eigen::VectorXd Strain = cipd.Apply(dofU, Nabla::Strain());
      return Strain;
  }

  Eigen::VectorXd Stress(const CellIpData &cipd)
  {
      Eigen::VectorXd stress = mC * Strain(cipd) - mE.transpose() * EField(cipd);
      return stress;
  }

  Eigen::VectorXd DField(const CellIpData &cipd)
  {
      Eigen::VectorXd DField = mP * EField(cipd) + mE * Strain(cipd);
      return DField;
  }


  DofVector<double> Gradient(const CellIpData &cipd) {

    Eigen::MatrixXd Belectrical = cipd.B(dofV, Nabla::Gradient());
    Eigen::MatrixXd Bstrain = cipd.B(dofU, Nabla::Strain());

    DofVector<double> gradient;

    gradient[dofV] =-Belectrical.transpose() * DField(cipd);
    gradient[dofU] = Bstrain.transpose() * Stress(cipd);

    return gradient;
  }

  DofMatrix<double> Hessian2(const CellIpData &cipd) {

    Eigen::MatrixXd Belectrical = cipd.B(dofV, Nabla::Gradient());
    Eigen::MatrixXd Bstrain = cipd.B(dofU, Nabla::Strain());

    DofMatrix<double> hessian2;

    hessian2(dofU, dofU) = Bstrain.transpose() * mC * Bstrain;
    hessian2(dofU, dofV) = Bstrain.transpose() * mE.transpose() * Belectrical;
    hessian2(dofV, dofU) = -Belectrical.transpose() * mE * Bstrain;
    hessian2(dofV, dofV) = Belectrical.transpose() * mP * Belectrical;

    return hessian2;
  }

  DofVector<double> NeumannLoadElectrical(const CellIpData &cipd, double Dn) {

    Eigen::MatrixXd N = cipd.N(dofV);
    DofVector<double> load;
    load[dofV] =  -N.transpose() * Dn;
    return load;
  }

  DofVector<double>
  NeumannLoadElectrical(const CellIpData &cipd,
                        std::function<double(Eigen::VectorXd)> Dn) {

    Eigen::MatrixXd N = cipd.N(dofV);
    DofVector<double> load;
    load[dofV] =  -N.transpose() * Dn(cipd.GlobalCoordinates());
    return load;
  }

  DofVector<double> NeumannLoadMechanical(const CellIpData &cipd,
                                          Eigen::Vector3d traction) {

    Eigen::MatrixXd N = cipd.N(dofU);
    DofVector<double> load;
    load[dofU] = N.transpose() * traction;
    return load;
  }

  DofVector<double> NeumannLoadMechanical(
      const CellIpData &cipd,
      std::function<Eigen::Vector3d(Eigen::VectorXd)> traction) {

    Eigen::MatrixXd N = cipd.N(dofU);
    DofVector<double> load;
    load[dofU] = N.transpose() * traction(cipd.GlobalCoordinates());
    return load;
  }

  std::vector<DofType> GetDofs()
  {
      return std::vector<DofType> {dofU, dofV};
  }

  DofType dofU;
  DofType dofV;

private:
  Eigen::MatrixXd mC;
  Eigen::MatrixXd mE;
  Eigen::MatrixXd mP;
};
} /* Integrand */
} /* NuTo */
