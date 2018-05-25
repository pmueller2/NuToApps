#include "nuto/base/Group.h"

#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTriangle.h"

#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/Cell.h"
#include "nuto/mechanics/cell/SimpleAssembler.h"

#include "nuto/mechanics/tools/CellStorage.h"
#include "nuto/mechanics/tools/TimeDependentProblem.h"
#include "omp.h"

using namespace NuTo;

int main() {
  MeshFem mesh = UnitMeshFem::CreateTriangles(2, 2);

  int omp_get_thread_num();

  DofType d("Displacements", 2);
  ScalarDofType temp("Temperature");

  AddDofInterpolation(&mesh, temp);
  AddDofInterpolation(&mesh, d);

  IntegrationTypeTriangle integrationType(2);

  CellStorage cellStorage;

  auto cells = cellStorage.AddCells(mesh.ElementsTotal(), integrationType);

  TimeDependentProblem eqs(&mesh);

  auto stiffness = [&](const CellIpData &cellIpData, double, double) {
    DofMatrix<double> stiffness;
    auto B = cellIpData.B(temp, Nabla::Gradient());
    stiffness(temp, temp) = B.transpose() * B;

    auto B2 = cellIpData.B(d, Nabla::Strain());
    stiffness(d, d) = B2.transpose() * B2;

    // grrr
    stiffness(temp, d) = Eigen::MatrixXd::Zero(B.cols(), B2.cols());
    stiffness(d, temp) = Eigen::MatrixXd::Zero(B2.cols(), B.cols());

    return stiffness;
  };

  eqs.AddHessian0Function(cells, stiffness);

  Constraint::Constraints bcs;
  auto u0 = eqs.RenumberDofs(bcs, {d, temp}, DofVector<double>());
  DofVector<double> dofs = u0;

  auto K = eqs.Hessian0(dofs, {d, temp}, 0.0, 0.0);

  return 0;
}
