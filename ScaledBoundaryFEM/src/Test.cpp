#include <iostream>

#include "nuto/mechanics/interpolation/InterpolationQuadQuadratic.h"
#include "nuto/mechanics/mesh/MeshFemDofConvert.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include "nuto/mechanics/integrationtypes/IntegrationTypeTensorProduct.h"

#include "nuto/mechanics/cell/Cell.h"

#include "nuto/mechanics/constraints/Constraints.h"

#include "nuto/mechanics/cell/SimpleAssembler.h"

#include "nuto/mechanics/tools/CellStorage.h"

#include "nuto/visualize/AverageHandler.h"
#include "nuto/visualize/Visualizer.h"

using namespace NuTo;

int main(int argc, char *argv[]) {
  std::cout << "Test" << std::endl;

  MeshFem mesh = UnitMeshFem::CreateQuads(5, 5);
  std::cout << "Expected: 25  " << mesh.Elements.Size() << std::endl;

  DofType dof("Displacements", 2);

  InterpolationQuadQuadratic ipol;

  AddDofInterpolation(&mesh, dof, ipol);

  int nCoords =
      mesh.Elements[0].CoordinateElement().Interpolation().GetNumNodes();
  int nDisplacements =
      mesh.Elements[0].DofElement(dof).Interpolation().GetNumNodes();

  IntegrationTypeTensorProduct<2> integr(1, eIntegrationMethod::GAUSS);

  //  int cellId = 0;
  //  Cell cell0(mesh.Elements[0], integr, cellId);

  //  Group<CellInterface> cellGroup;
  //  cellGroup.Add(cell0);

  CellStorage cells;
  Group<CellInterface> cellGroup = cells.AddCells(mesh.ElementsTotal(), integr);

  Constraint::Constraints constraints;

  DofInfo dofInfo = DofNumbering::Build(mesh.NodesTotal(dof), dof, constraints);

  SimpleAssembler asmbl(dofInfo);

  auto f = [dof](const CellIpData &cipd) {
    DofMatrix<double> matrix;

    // Eigen::MatrixXd B = cipd.B(dof,Nabla::???)

    return matrix;
  };

  asmbl.BuildMatrix(cellGroup, {dof}, f);

  Visualize::Visualizer vis(cellGroup, Visualize::AverageHandler());
  vis.DofValues(dof);
  vis.WriteVtuFile("Test.vtu");
}
