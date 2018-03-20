#pragma once

#include "nuto/mechanics/elements/ElementCollection.h"
#include "nuto/base/Group.h"
#include <functional>

namespace NuTo
{
namespace Tools
{

void SetValues(Group<ElementCollectionFem>& elements, DofType dof, std::function<double(Eigen::VectorXd)> func)
{
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        elmDof.GetNode(i).SetValue(0,func(Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i))));
      }
    }
}

void SetValues(Group<ElementCollectionFem>& elements, DofType dof, std::function<Eigen::VectorXd(Eigen::VectorXd)> func)
{
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        elmDof.GetNode(i).SetValues(func(Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i))));
      }
    }
}

void SetValues(Group<ElementCollectionFem>& elements, DofType dof, std::function<double(double)> func)
{
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        elmDof.GetNode(i).SetValue(0,func(Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i))[0]));
      }
    }
}

template<int TDim>
std::map<NodeSimple *, Eigen::Matrix<double,TDim,1>> GetNodeCoordinateMap(Group<ElementCollectionFem>& elements, DofType dof)
{
    std::map<NodeSimple *, Eigen::Matrix<double,TDim,1>> nodeCoordinateMap;
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        nodeCoordinateMap[&(elmDof.GetNode(i))] =
            Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
      }
    }
    return nodeCoordinateMap;
}

} /* Tools */
} /* NuTo */

