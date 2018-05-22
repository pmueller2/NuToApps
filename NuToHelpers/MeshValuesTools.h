#pragma once

#include "nuto/mechanics/elements/ElementCollection.h"
#include "nuto/mechanics/mesh/MeshFem.h"
#include "nuto/base/Group.h"
#include <functional>

namespace NuTo
{
namespace Tools
{

void SetValues(Group<ElementCollectionFem>& elements, DofType dof, std::function<double(Eigen::VectorXd)> func, int instance = 0)
{
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        elmDof.GetNode(i).SetValue(0,func(Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i))),instance);
      }
    }
}

void SetValues(Group<ElementCollectionFem>& elements, DofType dof, std::function<Eigen::VectorXd(Eigen::VectorXd)> func, int instance = 0)
{
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        elmDof.GetNode(i).SetValues(func(Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i))),instance);
      }
    }
}

void SetValues(Group<ElementCollectionFem>& elements, DofType dof, std::function<double(double)> func, int instance = 0)
{
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmCoord = elmColl.CoordinateElement();
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
        elmDof.GetNode(i).SetValue(0,func(Interpolate(elmCoord, elmDof.Interpolation().GetLocalCoords(i))[0]),instance);
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

std::map<NodeSimple *, std::vector<ElementCollectionFem *> > GetNodeElementMap(Group<ElementCollectionFem>& elements, DofType dof)
{
    std::map<NodeSimple *, std::vector<ElementCollectionFem *> > nodeElementMap;
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elmDof = elmColl.DofElement(dof);
      for (int i = 0; i < elmDof.Interpolation().GetNumNodes(); i++) {
          if (nodeElementMap.find(&(elmDof.GetNode(i))) == nodeElementMap.end())
          {
              nodeElementMap[&(elmDof.GetNode(i))] = std::vector<ElementCollectionFem *>{&elmColl};
          } else
          {
              nodeElementMap.at(&(elmDof.GetNode(i))).push_back(&elmColl);
          }
      }
    }
    return nodeElementMap;
}

std::map<NodeSimple *, std::vector<ElementCollectionFem *> > GetNodeCoordinateElementMap(const Group<ElementCollectionFem>& elements)
{
    std::map<NodeSimple *, std::vector<ElementCollectionFem *> > nodeElementMap;
    for (NuTo::ElementCollectionFem &elmColl : elements) {
      NuTo::ElementFem &elm = elmColl.CoordinateElement();
      for (int i = 0; i < elm.Interpolation().GetNumNodes(); i++) {
          if (nodeElementMap.find(&(elm.GetNode(i))) == nodeElementMap.end())
          {
              nodeElementMap[&(elm.GetNode(i))] = std::vector<ElementCollectionFem *>{&elmColl};
          } else
          {
              nodeElementMap.at(&(elm.GetNode(i))).push_back(&elmColl);
          }
      }
    }
    return nodeElementMap;
}

Eigen::VectorXd GetLocalCoordinatesFromGlobal(Eigen::VectorXd globalCoords, ElementFem& elm, int maxNumIterations = 20, double tol = 1e-7)
{
    auto& ipol = elm.Interpolation();
    int spaceDim = globalCoords.size();
    Eigen::MatrixXd nodeVals(ipol.GetNumNodes(),spaceDim);
    for (int i=0; i<nodeVals.rows();i++)
    {
        nodeVals.row(i) = elm.GetNode(i).GetValues().transpose();
    }
    Eigen::VectorXd localCoords = Eigen::VectorXd::Zero(spaceDim);
    int numIterations = 0;
    double correctorNorm = 1.;
    while ((correctorNorm > tol) && (numIterations < maxNumIterations))
    {
        Eigen::MatrixXd der = ipol.GetDerivativeShapeFunctions(localCoords);
        Eigen::MatrixXd jac = nodeVals.transpose() * der;
        Eigen::VectorXd res = globalCoords - Interpolate(elm,localCoords);
        Eigen::VectorXd dX = jac.inverse() * res;
        correctorNorm = dX.norm();
        Eigen::VectorXd xiNew = localCoords + dX;
        localCoords = xiNew;
        numIterations++;
    }
    if (numIterations == maxNumIterations)
        throw Exception(__PRETTY_FUNCTION__,"Exceeded number of iterations");
    return localCoords;
}

Eigen::VectorXd GetLocalCoordinates2Din3D(Eigen::VectorXd globalCoords, ElementFem& elm, int maxNumIterations = 20, double tol = 1e-7)
{
    auto& ipol = elm.Interpolation();
    int spaceDim = globalCoords.size();
    int localDim = elm.Interpolation().GetLocalCoords(0).size();

    if ((spaceDim != 3) && (localDim != 2))
    {
        throw Exception(__PRETTY_FUNCTION__,"Method is intended for 2D in 3D elements.");
    }

    Eigen::MatrixXd nodeVals(ipol.GetNumNodes(),spaceDim);
    for (int i=0; i<nodeVals.rows();i++)
    {
        nodeVals.row(i) = elm.GetNode(i).GetValues().transpose();
    }
    Eigen::VectorXd localCoords = Eigen::VectorXd::Zero(spaceDim);
    int numIterations = 0;
    double correctorNorm = 1.;
    while ((correctorNorm > tol) && (numIterations < maxNumIterations))
    {
        Eigen::MatrixXd der = ipol.GetDerivativeShapeFunctions(localCoords);
        Eigen::MatrixXd jac(spaceDim,spaceDim);
        Eigen::MatrixXd jacTmp = nodeVals.transpose() * der;
        for (int i=0; i<localDim; i++)
        {
            jac.col(i) = jacTmp.col(i);
        }
        Eigen::Vector3d colA = jacTmp.col(0);
        Eigen::Vector3d colB = jacTmp.col(1);
        Eigen::Vector3d normal = (colA.cross(colB)).normalized();
        jac.col(localDim) = normal;
        Eigen::VectorXd res = globalCoords - Interpolate(elm,localCoords);
        Eigen::VectorXd dX = jac.inverse() * res;
        correctorNorm = dX.norm();
        Eigen::VectorXd xiNew = localCoords + dX;
        localCoords = xiNew;
        numIterations++;
    }
    if (numIterations == maxNumIterations)
        throw Exception(__PRETTY_FUNCTION__,"Exceeded number of iterations");
    return localCoords;
}

class Interpolator
{
public:
    Interpolator(Eigen::MatrixXd coordinates, const Group<ElementCollectionFem> elements) : mElements(elements), mCoordinates(coordinates)
    {
        int numCoords = coordinates.rows();

        Group<NodeSimple> allCoordNodes;
          for (auto& element : mElements)
              for (int iNode = 0; iNode < element.CoordinateElement().Interpolation().GetNumNodes(); ++iNode)
                  allCoordNodes.Add(element.CoordinateElement().GetNode(iNode));

        auto coordinateNodeElementMap = Tools::GetNodeCoordinateElementMap(mElements);

        // Set up a list of corresponding localCoordinates and elements
        localOutputCoordinates.resize(numCoords);
        outputElements.resize(numCoords);

        for (int i = 0; i < numCoords; i++) {
          Eigen::VectorXd globalCoord = coordinates.row(i);
          double distance = INFINITY;
          NodeSimple *nearestNode;
          for (NodeSimple &nd : allCoordNodes) {
            if ((nd.GetValues() - globalCoord).norm() < distance) {
              nearestNode = &nd;
              distance = (nd.GetValues() - globalCoord).norm();
            }
          }
          auto nearbyElements = coordinateNodeElementMap.at(nearestNode);
          for (ElementCollectionFem *elm : nearbyElements) {
            Eigen::VectorXd localCoord = Tools::GetLocalCoordinatesFromGlobal(
                globalCoord, elm->CoordinateElement());
            if (elm->GetShape().IsWithinShape(localCoord)) {
              outputElements[i] = elm;
              localOutputCoordinates[i] = localCoord;
              break;
            }
          }
        }
    }

    Eigen::VectorXd GetValue(int i, DofType dof)
    {
        return Interpolate(outputElements[i]->DofElement(dof),localOutputCoordinates[i]);
    }

private:
    const Group<ElementCollectionFem>& mElements;
    Eigen::MatrixXd mCoordinates;
    std::vector<Eigen::VectorXd> localOutputCoordinates;
    std::vector<ElementCollectionFem *> outputElements;
};

} /* Tools */
} /* NuTo */

