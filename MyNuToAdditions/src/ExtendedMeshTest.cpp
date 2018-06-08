#include "nuto/mechanics/interpolation/InterpolationTriangleLinear.h"
#include "nuto/mechanics/interpolation/InterpolationTrussLinear.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

using namespace NuTo;

static auto ipolEdge = InterpolationTrussLinear();
static auto ipolTriangle = InterpolationTriangleLinear();

Group<ElementFem> CreateEdges(ElementFem &elm, MeshFem &mesh) {
  Group<ElementFem> edges;

  auto nd0 = elm.GetNode(0);
  auto nd1 = elm.GetNode(1);
  edges.Add((mesh.Elements.Add({{{nd0, nd1}, ipolEdge}})).CoordinateElement());
  return edges;
}

Group<ElementFem> CreateFaces(ElementFem &elm, MeshFem &mesh) {
  Group<ElementFem> edges;

  auto nd0 = elm.GetNode(0);
  auto nd1 = elm.GetNode(1);
  auto nd2 = elm.GetNode(2);
  edges.Add((mesh.Elements.Add({{{nd0, nd1, nd2}, ipolTriangle}}))
                .CoordinateElement());
  return edges;
}

void Test1() {
  // Generate a simple mesh
  MeshFem mesh = UnitMeshFem::CreateBricks(2, 3, 4);

  // Save current nodes and elements
  auto allNodes = mesh.NodesTotal();
  auto volumeElements = mesh.ElementsTotal();

  Group<ElementFem> edgeElements;
  Group<ElementFem> faceElements;

  // Now generate edge and face elements
  for (ElementCollectionFem &elmColl : volumeElements) {
    auto edges = CreateEdges(elmColl.CoordinateElement(), mesh);
    Unite(edgeElements, edges);
    auto faces = CreateFaces(elmColl.CoordinateElement(), mesh);
    Unite(faceElements, faces);
  }

  // Doing it this way means that one only has information flow
  // like so:
  //                Now            Save edge and face info in loop
  //
  //           Volume -> Nodes      -> Faces,Edges
  //           Face   -> Nodes      (-> Edges)
  //           Edge   -> Nodes
  //
  // How to avoid duplicates?
  // -> Think of a useful ordering
  //      (edges by first node, then by second
  //       faces by numNodes, then first node, then second, etc.
}

void Test2() { MeshFem mesh = UnitMeshFem::CreateBricks(2, 3, 4); }
int main() {
  // Test1();
  Test2();
}
