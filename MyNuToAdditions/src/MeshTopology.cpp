#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include <iostream>

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

struct Volume {
  Volume(std::vector<int> nodes, const Shape &s)
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
      auto topoElement = Volume(elmNodes, elm.GetShape());
      mElements.push_back(topoElement);
      // Edges
      Eigen::MatrixXi elementEdges;
      elementEdges = HexGetEdges(elmNodes);
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

  std::vector<Volume> mElements;
  std::vector<Edge> mEdges;

private:
  MeshFem &mesh;
  std::map<NodeSimple *, int> mNodes;
  Eigen::MatrixXd mEdgeConnectivity;
};

int main(int argc, char *argv[]) {

  MeshFem mesh = UnitMeshFem::CreateBricks(2, 3, 4);

  std::cout << "Generate mesh topology." << std::endl;

  MeshTopology mTopo(mesh);
}
