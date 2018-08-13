#include "nuto/mechanics/cell/CellIpData.h"
#include "nuto/mechanics/cell/DifferentialOperators.h"
#include "nuto/mechanics/cell/SimpleAssembler.h"
#include "nuto/mechanics/constitutive/LinearElastic.h"
#include "nuto/mechanics/dofs/DofNumbering.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"
#include "nuto/mechanics/interpolation/InterpolationBrickLobatto.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"
#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/base/Timer.h"

#include <functional>
#include <iostream>

using namespace NuTo;

/* Attempt to benchmark and optimize the time
 * needed to calculate the right hand side of
 * a simple wave problem
 */
int main(int argc, char *argv[]) {
  int nX = 3;
  int nY = 3;
  int nZ = 3;
  int order = 2;

  double E = 200.0e9;
  double nu = 0.3;
  double rho = 8000.;

  Laws::LinearElastic<3> steel(E, nu);

  MeshFem mesh;
  {
    Timer timer("CreateMesh");
    mesh = UnitMeshFem::CreateBricks(nX, nY, nZ);
  }
  std::cout << std::flush;
  auto allElements = mesh.ElementsTotal();

  DofType dof("displacement", 3);
  InterpolationBrickLobatto ipol(order);
  {
    Timer timer("AddDofInterpolation");
    AddDofInterpolation(&mesh, dof, allElements, ipol);
  }
  std::cout << std::flush;

  auto allCoordinateNodes = mesh.NodesTotal();
  auto allDofNodes = mesh.NodesTotal(dof);

  Constraint::Constraints constraint;

  DofInfo dofInfo = DofNumbering::Build(allDofNodes, dof, constraint);
  SimpleAssembler asmbl(dofInfo);

  IntegrationTypeTensorProduct<3> intType(order + 1,
                                          eIntegrationMethod::LOBATTO);

  CellStorage cells;
  auto cellGroup = cells.AddCells(allElements, intType);

  auto rightHandSide = [dof, &steel](const CellIpData &cipd) {

    DofVector<double> gradient;

    Eigen::MatrixXd B = cipd.B(dof, Nabla::Strain());
    gradient[dof] =
        B.transpose() *
        steel.Stress(cipd.Apply(dof, Nabla::Strain()), 0., cipd.Ids());

    return gradient;
  };

  Eigen::VectorXd localGradient(ipol.GetNumNodes() * dof.GetNum());
  localGradient.setOnes();

  DofVector<double> localGradientDof;
  localGradientDof[dof] = localGradient;

  auto constantRHS = [dof, &localGradient](const CellIpData &cipd) {

    DofVector<double> gradient;
    gradient[dof] = localGradient;
    return gradient;
  };

  DofVector<double> globalDofVector;
  {
    Timer timer("VectorAssembly");
    globalDofVector = asmbl.BuildVector(cellGroup, {dof}, rightHandSide);
  }

  // ****************************

  //  {
  //    Timer timer("ThrowOnZeroDofNumbering");

  //    if (not dofInfo.numIndependentDofs.Has(dof))
  //      throw Exception(
  //          "[NuTo::SimpleAssembler]",
  //          "You did not provide a dof numbering for DofType " + dof.GetName()
  //          +
  //              ". Please do so by SimpleAssembler::calling
  //              SetDofInfo(...).");
  //  }

  //  DofVector<double> gradient;
  //  {
  //    Timer timer("Properly resizedDofVector");
  //    gradient[dof].setZero(dofInfo.numIndependentDofs[dof] +
  //                          dofInfo.numDependentDofs[dof]);
  //  }

  //  {
  //    Timer timer("MyVectorAssembly");

  //#pragma omp parallel
  //    {
  //      DofVector<double> threadlocalgradient = gradient;
  //#pragma omp for nowait
  //      for (auto cellit = cellGroup.begin(); cellit < cellGroup.end();
  //           cellit++) {
  //        //        const DofVector<double> cellGradient =
  //        //        cellit->Integrate(rightHandSide);
  //        // roll out Cell Integrate
  //        DofVector<double> cellGradient;

  //        CellData cellData(cellit->mElements, cellit->mId);
  //        for (int iIP = 0;
  //             iIP < cellit->mIntegrationType.GetNumIntegrationPoints();
  //             ++iIP) {
  //          auto ipCoords =
  //              cellit->mIntegrationType.GetLocalIntegrationPointCoordinates(iIP);
  //          auto ipWeight =
  //              cellit->mIntegrationType.GetIntegrationPointWeight(iIP);
  //          const Jacobian &jacobian = cellit->mJacobianMemo.Get(ipCoords);
  //          CellIpData cellipData(cellData, jacobian, ipCoords, iIP);
  //          cellGradient += rightHandSide(cellipData) * jacobian.Det() *
  //          ipWeight;
  //        }
  //        // end Cell Integrate
  //        std::vector<DofType> intersection;
  //        for (DofType d1 : cellGradient.DofTypes())
  //          for (DofType d2 : {dof})
  //            if (d1.Id() == d2.Id()) {
  //              intersection.push_back(d1);
  //              continue;
  //            }

  //        for (DofType dof : intersection) {
  //          Eigen::VectorXi numberingDof = cellit->DofNumbering(dof);
  //          const Eigen::VectorXd &cellGradientDof = cellGradient[dof];
  //          for (int i = 0; i < numberingDof.rows(); ++i)
  //            threadlocalgradient[dof](numberingDof[i]) += cellGradientDof[i];
  //        }
  //      }
  //#pragma omp critical
  //      gradient += threadlocalgradient;
  //    }
  //  }

  //  // ****************************

  //  std::cout << std::flush;
}
