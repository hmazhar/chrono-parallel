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
// Authors: Hammad Mazhar, Daniel Melanz
// =============================================================================
//
// Implementation of APGD that is exactly like the thesis
// =============================================================================

#ifndef CHSOLVERAPGDREF_H
#define CHSOLVERAPGDREF_H

#include "chrono_parallel/solver/ChSolverParallel.h"

namespace chrono {

class CH_PARALLEL_API ChSolverAPGDREF: public ChSolverParallel {
public:
  ChSolverAPGDREF() : ChSolverParallel() {}
  ~ChSolverAPGDREF() {}

  void Solve() {
    if (data_container->num_constraints == 0) {
      return;
    }
    data_container->system_timer.start("ChSolverParallel_Solve");
    data_container->measures.solver.total_iteration +=
      SolveAPGDREF(max_iteration,
                   data_container->num_constraints,
                   data_container->host_data.R,
                   data_container->host_data.gamma);
    data_container->system_timer.stop("ChSolverParallel_Solve");
  }

  // Solve using the APGD method
  uint SolveAPGDREF(const uint max_iter,                    // Maximum number of iterations
                    const uint size,                        // Number of unknowns
                    const DenseVector& r,    // Rhs vector
                    DenseVector& gamma       // The vector of unknowns
                    );

  // Compute the residual for the solver
  real Res4(DenseVector& gamma,
            const DenseVector& r,
            DenseVector& tmp);

  // APGD specific vectors
  DenseVector gamma_hat;
  DenseVector gammaNew, g, y, yNew, tmp;
};

}

#endif
