#include "nuto/mechanics/mesh/UnitMeshFem.h"

#include <iostream>

using namespace NuTo;

int main(int argc, char *argv[]) {

  MeshFem mesh = UnitMeshFem::CreateLines(5);
  std::cout << mesh.NodesTotal().Size() << std::endl;
}
