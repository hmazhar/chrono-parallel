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
// Authors: Daniel Melanz
// =============================================================================
//
// This file contains an implementation of PDIP.
// TODO: Replace the Dinv matrix with a function
// TODO: Get rid of as many matrices as possible (diaglambda could be replaced)
// TODO: Use better linear solve than cg
// =============================================================================

#ifndef CHSOLVERMOSEK_H
#define CHSOLVERMOSEK_H

#include "chrono_parallel/solver/ChSolverParallel.h"

namespace chrono {

class CH_PARALLEL_API ChSolverMosek : public ChSolverParallel {
 public:
  ChSolverMosek();
  ~ChSolverMosek() {}

  void Solve() {
    if (data_container->num_constraints == 0) {
      return;
    }
    data_container->system_timer.start("ChSolverParallel_Solve");

    data_container->measures.solver.total_iteration += SolveMosek(
        max_iteration, data_container->num_constraints, data_container->host_data.R, data_container->host_data.gamma);
    data_container->system_timer.stop("ChSolverParallel_Solve");
  }

  // Solve using the primal-dual interior point method
  uint SolveMosek(const uint max_iter,                  // Maximum number of iterations
                  const uint size,                      // Number of unknowns
                  const blaze::DynamicVector<real>& b,  // Rhs vector
                  blaze::DynamicVector<real>& x         // The vector of unknowns
                  );

  // PDIP specific vectors
  blaze::CompressedMatrix<real> D_T, M_invD;
};
}

#endif
