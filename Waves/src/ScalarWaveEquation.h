#pragma once

#include "boost/filesystem.hpp"
#include "nuto/mechanics/cell/CellInterface.h"
#include "nuto/mechanics/constraints/Constraints.h"
#include "nuto/mechanics/dofs/DofType.h"
#include "nuto/mechanics/integrationtypes/IntegrationTypeBase.h"
#include "nuto/mechanics/mesh/MeshFem.h"
#include <boost/ptr_container/ptr_vector.hpp>

//! Represents the scalar wave equation
//!
//! d2u/dt2 = div (C grad u) + f
//!
//! With boundary data Dirichlet u(Dirichlet) = u_D
//!                  or Neumann (C grad u) n = g_N
//!
//! And initial data u(0,x) = u0(x)
//!            and   v(0,x) = v0(x) (velocities)
//!
//! The medium parameter C can be a matrix
//! The functions u_D,g_N,f,C may depend on space and time
//!
//! Since the equation is linear its weak form may be written
//! as matrix equation:  M d2u/dt2 + Ku = f
//!
//! 2nd order ODE
//! ->  d2u/dt2 = M^{-1} (f - Ku)
//!
//! or
//!
//! 1st order ODE
//! ->  du/dt = v
//!     dv/dt = M^{-1} (f - Ku)
//!
//! In each time step the matrix vector multiplication Ku will be performed.
//! No element wise evaluation will be done.
class ScalarWaveEquation {
public:
  ScalarWaveEquation(NuTo::MeshFem &mesh);

  void SetDomain(NuTo::Group<NuTo::ElementCollectionFem> elements);
  void SetBoundary(NuTo::Group<NuTo::ElementCollectionFem> elements);

  //! Set dirichlet boundary (constant in time, space dependent) at elements
  void SetDirichletBoundary(NuTo::Group<NuTo::ElementCollectionFem> elements,
                            std::function<double(Eigen::VectorXd)> func);

  //! Set dirichlet boundary (constant in time and space) at elements
  void SetDirichletBoundary(NuTo::Group<NuTo::ElementCollectionFem> elements,
                            double val);

  //! Set dirichlet boundary (constant in time, space dependent) at nodes
  void SetDirichletBoundary(NuTo::Group<NuTo::NodeSimple> coordinateNodes,
                            std::function<double(Eigen::VectorXd)> func);

  //! Set dirichlet boundary (constant in time and space) at nodes
  void SetDirichletBoundary(NuTo::Group<NuTo::NodeSimple> coordinateNodes,
                            double val);

  void SetNeumannBoundary(NuTo::Group<NuTo::ElementCollectionFem> elements);

  //! Set interpolation and integration order
  //! adds Lobatto interpolations to all elements
  void SetOrder(int order);

  //! Set result directory (will be in results ...)
  void SetResultDirectory(std::string resultDirectory,
                          bool overwriteResultDirectory = true);

  void Solve(double simTime, double timeStep);

private:
  NuTo::MeshFem &mMesh;
  NuTo::Group<NuTo::ElementCollectionFem> mDomain;
  NuTo::Group<NuTo::ElementCollectionFem> mBoundary;
  NuTo::Group<NuTo::ElementCollectionFem> mNeumannBoundary;
  NuTo::Group<NuTo::CellInterface> mDomainCells;
  NuTo::Group<NuTo::CellInterface> mNeumannBoundaryCells;
  NuTo::Group<NuTo::NodeSimple> mDofNodes;

  NuTo::DofType mDof;
  int mInterpolationOrder;
  boost::ptr_vector<NuTo::CellInterface> mCells;
  std::vector<std::unique_ptr<NuTo::IntegrationTypeBase>> mIntTypes;
  boost::filesystem::path mResultDirectoryFull;
  NuTo::Constraint::Constraints mConstraints;
};
