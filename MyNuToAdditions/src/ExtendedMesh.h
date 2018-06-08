#pragma once

#include "nuto/mechanics/elements/ElementFem.h"
#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/base/Group.h"


#include "MyInterpolationTriangleQuadratic.h"

namespace NuTo
{

namespace Mesh
{

//! @brief Adds edge elements to mesh
//! @param rMesh fem mesh, return argument with r and weird pointer syntax to make it clear
//! @param rMesh elm, element whose edges are added
Group<ElementCollectionFem> AddEdgeElements(MeshFem* rMesh, ElementFem& elm)
{
    std::vector<std::vector<int>> edgeNodeIds;
    // edgeNodeIds = elm.Interpolation().GetEdgeNodeIds(); -> Has to be implemented for all interpolations

    // Initialize with some values (2nd order triangle):
    int numEdges = 3;
    int numNodesPerEdge = 3;

    edgeNodeIds.push_back({0,1,2}); // Edge1 (second order)
    edgeNodeIds.push_back({2,3,4}); // Edge2 (second order)
    edgeNodeIds.push_back({4,5,0}); // Edge3 (second order)

    // Now add the corresponding interpolation
    // rMesh->CreateInterpolation(elm.Interpolation().GetEdgeInterpolation()) -> Has to be implemented for all interpolations
    InterpolationTriangleQuadratic ipol;
    rMesh->CreateInterpolation(ipol);

    // Now add elements
    std::vector<NodeSimple*> nodes0;
    for (int i : edgeNodeIds[0])
    {
        nodes0.push_back(&(elm.GetNode(i)));
    }
    auto& e0 = rMesh->Elements.Add({{nodes0,ipol}});


    std::vector<NodeSimple*> nodes1;
    for (int i : edgeNodeIds[1])
    {
        nodes1.push_back(&(elm.GetNode(i)));
    }
    auto& e1 = rMesh->Elements.Add({{nodes1,ipol}});


    std::vector<NodeSimple*> nodes2;
    for (int i : edgeNodeIds[2])
    {
        nodes2.push_back(&(elm.GetNode(i)));
    }
    auto& e2 = rMesh->Elements.Add({{nodes2,ipol}});

    Group<ElementCollectionFem> edgeElements;
    edgeElements.Add(e0);
    edgeElements.Add(e1);
    edgeElements.Add(e2);
    return edgeElements;
}

//! @brief Adds edge elements to mesh, no duplicates
//! @param rMesh fem mesh, return argument with r and weird pointer syntax to make it clear
//! @param rMesh elms, elements whose edges are added
Group<ElementCollectionFem> AddEdgeElements(MeshFem* rMesh, Group<ElementFem>& elm)
{
    // To be implemented. Should not introduce duplicates
}

Group<ElementFem> AddFaceElements(MeshFem* rMesh, const ElementFem& elm)
{

}

} /* Mesh */

} /* Nuto */
