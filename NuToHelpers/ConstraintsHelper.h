#pragma once

#include "nuto/mechanics/elements/ElementCollection.h"
#include "nuto/mechanics/constraints/ConstraintCompanion.h"

#include <set>

namespace NuTo
{
namespace Constraint
{

Eigen::VectorXi GetDependentGlobalDofNumbering(Constraints constraints, DofType dof, int numDofs)
{
    Eigen::VectorXi dependentGlobalNumbering(constraints.GetNumEquations(dof));
    for (int i = 0; i < constraints.GetNumEquations(dof); i++) {
      int globalDofNumber =
          constraints.GetEquation(dof, i).GetDependentDofNumber();
      if (globalDofNumber == -1 /* should be NodeSimple::NOT_SET */)
        throw Exception(__PRETTY_FUNCTION__,
                        "There is no dof numbering for a node in equation" +
                            std::to_string(i) + ".");
      if (globalDofNumber >= numDofs)
        throw Exception(__PRETTY_FUNCTION__,
                        "The provided dof number of the dependent term exceeds "
                        "the total number of dofs in equation " +
                            std::to_string(i) + ".");
      dependentGlobalNumbering[i] = globalDofNumber;
    }
    return dependentGlobalNumbering;
}

Eigen::VectorXi GetIndependentGlobalDofNumbering(Constraints constraints, DofType dof, int numDofs)
{
    Eigen::VectorXi dep = GetDependentGlobalDofNumbering(constraints, dof, numDofs);
    std::sort(dep.data(),dep.data()+dep.size());

    auto isConstrained = [&](int val){
        auto first =dep.data();
        auto last = dep.data() + dep.size();
        auto it = std::lower_bound(first,last,val);
        if ( (it != last) && (*it == val) )
            return true;
        return false;
    };

    Eigen::VectorXi indep(numDofs - constraints.GetNumEquations(dof));
    int count = 0;
    for (int i=0; i<numDofs; i++)
    {
        if (isConstrained(i))
            continue;
        indep(count) = i;
        count++;
    }
    return indep;
}

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

