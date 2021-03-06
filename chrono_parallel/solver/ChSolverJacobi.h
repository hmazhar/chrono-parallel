// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Hammad Mazhar
// =============================================================================
//
// Implementation of the Jacobi iterative solver.
// =============================================================================

#ifndef CHSOLVERJACOBI_H
#define CHSOLVERJACOBI_H

#include "chrono_parallel/solver/ChSolverParallel.h"

namespace chrono {

class CH_PARALLEL_API ChSolverJacobi : public ChSolverParallel {
 public:
  ChSolverJacobi() : ChSolverParallel() {}
  ~ChSolverJacobi() {}

  void Solve() {
    if (data_manager->num_constraints == 0) {
      return;
    }

    data_manager->measures.solver.total_iteration += SolveJacobi(
        max_iteration, data_manager->num_constraints, data_manager->host_data.R, data_manager->host_data.gamma);
  }

  // Solve using the Jacobi method
  uint SolveJacobi(const uint max_iter,            // Maximum number of iterations
                   const uint size,                // Number of unknowns
                   DynamicVector<real>& b,  // Rhs vector
                   DynamicVector<real>& x   // The vector of unknowns
                   );

  custom_vector<real> r, p, Ap;
  DynamicVector<real> diagonal, ml_old,  N_gamma_old;
  DynamicVector<real> N_gamma_new, temp;
};
}

#endif
