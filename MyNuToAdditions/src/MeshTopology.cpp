#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include <iostream>

Eigen::MatrixXi HexGetEdges(std::vector<int> vertices) {
  assert(vertices.size() == 8);
  int numEdges = 12;
  Eigen::MatrixXi edges(numEdges, 2);
  edges(0, 0) = vertices[0];
  edges(0, 1) = vertices[1];

  edges(1, 0) = vertices[1];
  edges(1, 1) = vertices[2];

  edges(2, 0) = vertices[2];
  edges(2, 1) = vertices[3];

  edges(3, 0) = vertices[3];
  edges(3, 1) = vertices[0];

  edges(4, 0) = vertices[4];
  edges(4, 1) = vertices[5];

  edges(5, 0) = vertices[5];
  edges(5, 1) = vertices[6];

  edges(6, 0) = vertices[6];
  edges(6, 1) = vertices[7];

  edges(7, 0) = vertices[7];
  edges(7, 1) = vertices[4];

  edges(8, 0) = vertices[0];
  edges(8, 1) = vertices[4];

  edges(9, 0) = vertices[1];
  edges(9, 1) = vertices[5];

  edges(10, 0) = vertices[2];
  edges(10, 1) = vertices[6];

  edges(11, 0) = vertices[3];
  edges(11, 1) = vertices[7];
  return edges;
}

Eigen::MatrixXi TriangleGetEdges(std::vector<int> vertices) {
  assert(vertices.size() == 3);
  int numEdges = 3;
  Eigen::MatrixXi edges(numEdges, 2);
  edges(0, 0) = vertices[0];
  edges(0, 1) = vertices[1];

  edges(1, 0) = vertices[1];
  edges(1, 1) = vertices[2];

  edges(2, 0) = vertices[2];
  edges(2, 1) = vertices[0];
  return edges;
}

Eigen::MatrixXi QuadGetEdges(std::vector<int> vertices) {
  assert(vertices.size() == 4);
  int numEdges = 4;
  Eigen::MatrixXi edges(numEdges, 2);
  edges(0, 0) = vertices[0];
  edges(0, 1) = vertices[1];

  edges(1, 0) = vertices[1];
  edges(1, 1) = vertices[2];

  edges(2, 0) = vertices[2];
  edges(2, 1) = vertices[3];

  edges(3, 0) = vertices[3];
  edges(3, 1) = vertices[0];
  return edges;
}

using namespace NuTo;

struct Edge {
  Edge(int a, int b) : mStart(a), mEnd(b) {}

  int mStart;
  int mEnd;
};

struct Face {
  Face(std::vector<int> nodes, Shape &s) : mNodes(nodes), mShape(s.Enum()) {}

  std::vector<int> mNodes;
  eShape mShape;
};

struct Element {
  Element(std::vector<int> nodes, const Shape &s)
      : mNodes(nodes), mShape(s.Enum()) {}

  std::vector<int> mNodes;
  eShape mShape;
};

class MeshTopology {
public:
  MeshTopology(MeshFem &m) : mesh(m) {
    // Fill node map
    int nodeCounter = 0;
    for (NodeSimple &nd : mesh.NodesTotal()) {
      mNodes[&nd] = nodeCounter;
      ++nodeCounter;
    }
    // Fill topology
    int edgeCounter = 0;
    mEdgeConnectivity.resize(nodeCounter, nodeCounter);
    mEdgeConnectivity.setZero();
    for (ElementCollectionFem &elmColl : mesh.Elements) {
      ElementFem &elm = elmColl.CoordinateElement();
      // Elements
      std::vector<int> elmNodes;
      for (int i = 0; i < elm.GetNumNodes(); i++) {
        NodeSimple &nd = elm.GetNode(i);
        int ndId = mNodes.at(&nd);
        elmNodes.push_back(ndId);
      }
      auto topoElement = Element(elmNodes, elm.GetShape());
      mElements.push_back(topoElement);
      // Edges
      Eigen::MatrixXi elementEdges;
      switch (elm.GetShape().Enum()) {
      case eShape::Triangle: {
        elementEdges = TriangleGetEdges(elmNodes);
        break;
      }
      case eShape::Quadrilateral: {
        elementEdges = QuadGetEdges(elmNodes);
        break;
      }
      case eShape::Hexahedron: {
        elementEdges = HexGetEdges(elmNodes);
        break;
      }
      default:
        throw Exception(__PRETTY_FUNCTION__, "Shape edges not implemented");
      }
      for (int j = 0; j < elementEdges.rows(); j++) {
        int startNode = elementEdges(j, 0);
        int endNode = elementEdges(j, 1);
        if (mEdgeConnectivity(startNode, endNode) != 0)
          continue;
        if (mEdgeConnectivity(endNode, startNode) != 0) {
          mEdgeConnectivity(startNode, endNode) = -edgeCounter;
        } else {
          mEdgeConnectivity(startNode, endNode) = edgeCounter;
          mEdges.push_back(Edge(startNode, endNode));
          edgeCounter++;
        }
      }
    }
  }

  std::vector<Element> mElements;
  std::vector<Edge> mEdges;

private:
  MeshFem &mesh;
  std::map<NodeSimple *, int> mNodes;
  Eigen::MatrixXd mEdgeConnectivity;
};

int main(int argc, char *argv[]) {

  std::cout << "Generate a mesh." << std::endl;

  // MeshFem mesh = UnitMeshFem::CreateQuads(5, 7);
  MeshFem mesh = UnitMeshFem::CreateBricks(2, 3, 4);

  std::cout << "Generate mesh topology." << std::endl;

  MeshTopology mTopo(mesh);
}
