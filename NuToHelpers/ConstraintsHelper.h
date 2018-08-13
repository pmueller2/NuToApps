#pragma once

#include "nuto/mechanics/elements/ElementCollection.h"
#include "nuto/mechanics/constraints/ConstraintCompanion.h"

#include <set>

namespace NuTo
{
namespace Constraint
{

std::vector<Equation> SetDirichletBoundaryNodes(DofType dof,
        NuTo::Group<ElementCollectionFem>& elements,
        std::function<double(Eigen::VectorXd)> f)
{
    std::vector<Equation> equations;
    std::set<NodeSimple *> nodes;

    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        NodeSimple& nd = elmDof.GetNode(i);
        // If this node was already used: continue with next one
        if (nodes.find(&nd) != nodes.end())
            continue;
        nodes.insert(&nd);
        Eigen::VectorXd coords = Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        double value = f(coords);
        equations.push_back(Constraint::Value(nd, value));
      }
    }
    return equations;
}

std::vector<Equation> SetDirichletBoundaryNodes(DofType dof,
        NuTo::Group<ElementCollectionFem>& elements,
        std::function<Eigen::VectorXd(Eigen::VectorXd)> f)
{
    std::vector<Equation> equations;
    std::set<NodeSimple *> nodes;

    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        NodeSimple& nd = elmDof.GetNode(i);
        // If this node was already used: continue with next one
        if (nodes.find(&nd) != nodes.end())
            continue;
        nodes.insert(&nd);
        Eigen::VectorXd coords = Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i));
        Eigen::VectorXd value = f(coords);
        std::vector<eDirection> dir {eDirection::X,eDirection::Y,eDirection::Z};
        for (int i=0; i<value.size(); i++)
        {
            equations.push_back(Constraint::Component(nd, {dir[i]}, value[i])[0]);
        }
      }
    }
    return equations;
}

} /* Constraint */

Group<NodeSimple> GetNodes(const NuTo::Group<ElementCollectionFem> g, DofType dof)
{
    Group<NodeSimple> nodes;
    for (auto &elm : g) {
      auto &e = elm.DofElement(dof);
      for (int i = 0; i < e.GetNumNodes(); i++) {
        nodes.Add(e.GetNode(i));
      }
    }
    return nodes;
}

} /* NuTo */

