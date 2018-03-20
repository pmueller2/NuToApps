#include <iostream>

#include "boost/ptr_container/ptr_vector.hpp"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/CellInterface.h"

#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLobatto.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/visualize/Visualizer.h"
#include "nuto/visualize/VoronoiGeometries.h"
#include "nuto/visualize/VoronoiHandler.h"

#include "../../NuToHelpers/InterpolationTrussTrigonometric.h"

using namespace NuTo;

class Interpolator {
public:
  MeshFem mesh;
  DofType dof;
  boost::ptr_vector<CellInterface> cells;
  Group<CellInterface> cellGroup;
  IntegrationTypeBase &integrationType1D;
  std::function<Eigen::VectorXd(Eigen::VectorXd)> funcToInterpolate;

  Interpolator(InterpolationSimple &ipol, MeshFem msh,
               IntegrationTypeBase &integr,
               std::function<Eigen::VectorXd(Eigen::VectorXd)> func)
      : mesh(std::move(msh)), dof("Interpolated", 1), integrationType1D(integr),
        funcToInterpolate(func) {

    auto &interpolation = mesh.CreateInterpolation(ipol);
    AddDofInterpolation(&mesh, dof, interpolation);

    int cellId = 0;
    for (ElementCollection &element : mesh.Elements) {
      cells.push_back(new Cell(element, integrationType1D, cellId++));
    }
    for (CellInterface &c : cells) {
      cellGroup.Add(c);
    }

    DoInterpolate();
  }

  void DoInterpolate() {
    for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        Eigen::VectorXd coord =
            Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        elmDof.GetNode(i).SetValue(0, funcToInterpolate(coord)[0]);
      }
    };
  }

  double errorL2() {
    auto squareErrorFunc = [&](const CellIpData &cipd) {

      Eigen::VectorXd localVal = cipd.Value(dof);
      double error =
          localVal[0] - funcToInterpolate(cipd.GlobalCoordinates())[0];
      return (error * error);
    };

    double L2Error = 0.;
    for (CellInterface &cell : cellGroup) {
      L2Error += sqrt(cell.Integrate(squareErrorFunc));
    }
    L2Error /= cellGroup.Size();
    return L2Error;
  }

  double errorEnergy() {
    auto energyErrorFunc = [&](const CellIpData &cipd) {

      Eigen::VectorXd localVal = cipd.Apply(dof, Nabla::Gradient());
      double dx = 1.e-8;
      double estimatedDerivative = funcToInterpolate(
          cipd.GlobalCoordinates() + Eigen::VectorXd::Constant(1, dx))[0];
      estimatedDerivative -= funcToInterpolate(cipd.GlobalCoordinates())[0];
      estimatedDerivative /= dx;
      double error = localVal[0] - estimatedDerivative;
      return (error * error);
    };

    double EnergyError = 0.;
    for (CellInterface &cell : cellGroup) {
      EnergyError += sqrt(cell.Integrate(energyErrorFunc));
    }
    EnergyError /= cellGroup.Size();
    return EnergyError;
  }

  double errorMax() {
    double maxError = 0.;

    auto maxErrorFunc = [&](const CellIpData &cipd) {

      Eigen::VectorXd localVal = cipd.Value(dof);
      double error =
          localVal[0] - funcToInterpolate(cipd.GlobalCoordinates())[0];
      return (std::max(maxError, std::abs(error)));
    };

    for (CellInterface &cell : cellGroup) {
      double errorResult = cell.Integrate(maxErrorFunc);
      maxError = errorResult;
    }
    return maxError;
  }

  //! \brief maximum absolute distance to true solution at nodes
  double errorMaxNode() {

    double maxNodeError = 0.;
    for (NuTo::ElementCollectionFem &elmColl : mesh.Elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        Eigen::VectorXd coord =
            Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        double localError =
            elmDof.GetNode(i).GetValues()[0] - funcToInterpolate(coord)[0];
        maxNodeError = std::max(maxNodeError, std::abs(localError));
      }
    };
    return maxNodeError;
  }

  void plot(std::string filename) {
    NuTo::Visualize::Visualizer visualize(
        cellGroup,
        NuTo::Visualize::VoronoiHandler(NuTo::Visualize::VoronoiGeometryLine(
            20 * (integrationType1D.GetNumIntegrationPoints() + 1))));
    visualize.DofValues(dof);
    visualize.PointData(funcToInterpolate, "Exact");
    visualize.WriteVtuFile(filename + ".vtu");
  }
};

double f1(double x) { return 1. / (5. - 4. * cos(x * 2. * M_PI)); }

double f2(double x, double p = 2.) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  double y = 2. * z - 1.;
  return std::pow((std::abs(1. - std::pow(std::abs(y), p))), (1. / p));
}

double f3(double x) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  return z;
}

double f4(double x, double extent = 0.3) {
  double z;
  if (x < 0.) {
    z = 1. + fmod(x, 1.);
  } else {
    z = fmod(x, 1.);
  }
  z = 2. / extent * z - 1.;
  if ((z < -1.) || (z > 1.))
    return 0.;
  return (0.5 * (1. + cos(M_PI * z)));
}

void RunTests() {

  //************************************
  //**     Num Elements        *********
  //************************************

  int maxNumDofs = 50;
  // int numDofs = (ipol.GetNumNodes() - 1) + 1;
  int numDofs = 1;
  int numElms = 1;

  // for (int numElms = 1; numDofs < maxNumDofs; numElms++) {
  // for (int numElms = 1; numElms <= maxNumElms; numElms++) {
  for (int order = 1; numDofs < maxNumDofs; order++) {
    // int order = 49;
    // int maxNumElms = 1;
    IntegrationTypeTensorProduct<1> integr(10 * (order + 1),
                                           eIntegrationMethod::LOBATTO);
    // InterpolationTrussTrigonometric ipol(order);
    InterpolationTrussLobatto ipol(order);
    // InterpolationTrussLinear ipol;

    numDofs = (ipol.GetNumNodes() - 1) * numElms + 1;

    //************************************
    //**            Shifting     *********
    //************************************

    int numSteps = 100;

    double maxErrorL2 = 0.;
    double minErrorL2 = INFINITY;
    double maxErrorMax = 0.;
    double minErrorMax = INFINITY;

    for (int i = 0; i < numSteps; i++) {
      double shift = i * 1. / (numSteps - 1);
      Interpolator ip(ipol, UnitMeshFem::CreateLines(numElms), integr,
                      [&](Eigen::VectorXd x) {
                        return Eigen::VectorXd::Constant(1, f4(x[0] - shift));
                      });
      double errorL2current = ip.errorL2();
      double errorMaxcurrent = ip.errorMax();

      // std::cout << "Shift, L2, Max " << std::endl;
      //      std::cout << shift << "   ";
      //      std::cout << errorL2current << "   ";
      //      std::cout << errorMaxcurrent << std::endl;

      maxErrorL2 = std::max(maxErrorL2, errorL2current);
      minErrorL2 = std::min(minErrorL2, errorL2current);

      maxErrorMax = std::max(maxErrorMax, errorMaxcurrent);
      minErrorMax = std::min(minErrorMax, errorMaxcurrent);

      // ip.plot("f" + std::to_string(i));
    }

    // std::cout << "NumDofs, L2, Max " << std::endl;
    std::cout << numDofs << "  ";
    std::cout << maxErrorL2 << "  ";
    std::cout << minErrorL2 << "  ";
    std::cout << maxErrorMax << "  ";
    std::cout << minErrorMax << std::endl;
  }
}

void smallTest() {
  int order = 2;

  IntegrationTypeTensorProduct<1> integr(10 * (order + 1),
                                         eIntegrationMethod::LOBATTO);
  // InterpolationTrussTrigonometric ipol(order);
  InterpolationTrussLobatto ipol(order);

  Interpolator ip(ipol, UnitMeshFem::CreateLines(20), integr,
                  [&](Eigen::VectorXd x) {
                    return Eigen::VectorXd::Constant(1, f2(x[0]));
                  });

  ip.plot("Test");

  std::cout << ip.errorEnergy() << std::endl;
  std::cout << ip.errorL2() << std::endl;
  std::cout << ip.errorMax() << std::endl;
}

int main(int argc, char *argv[]) { smallTest(); }
