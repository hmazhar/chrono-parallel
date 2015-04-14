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

#ifndef CHSOLVERPDIP_H
#define CHSOLVERPDIP_H

#include "chrono_parallel/solver/ChSolverParallel.h"

namespace chrono {

class CH_PARALLEL_API ChSolverPDIP : public ChSolverParallel {
 public:
  ChSolverPDIP() : ChSolverParallel() {}
  ~ChSolverPDIP() {}

  void Solve() {
    if (data_manager->num_constraints == 0) {
      return;
    }
    data_manager->system_timer.start("ChSolverParallel_Solve");

    data_manager->measures.solver.total_iteration += SolvePDIP(
        max_iteration, data_manager->num_constraints, data_manager->host_data.R, data_manager->host_data.gamma);
    data_manager->system_timer.stop("ChSolverParallel_Solve");
  }

  // Solve using the primal-dual interior point method
  uint SolvePDIP(const uint max_iter,           // Maximum number of iterations
                 const uint size,               // Number of unknowns
                 const DynamicVector<real>& b,  // Rhs vector
                 DynamicVector<real>& x         // The vector of unknowns
                 );

  // Compute the residual for the solver
  real Res4(DynamicVector<real>& gamma, DynamicVector<real>& tmp);

  // Compute the Schur Complement Product, dst = N * src
  void SchurComplementProduct(DynamicVector<real>& src, DynamicVector<real>& dst);
  void initializeNewtonStepMatrix(DynamicVector<real>& gamma,
                                  DynamicVector<real>& lambda,
                                  DynamicVector<real>& f,
                                  const uint size);
  void initializeConstraintGradient(DynamicVector<real>& src, const uint size);
  void getConstraintVector(DynamicVector<real>& src, DynamicVector<real>& dst, const uint size);
  void updateConstraintGradient(DynamicVector<real>& src, const uint size);
  void updateNewtonStepMatrix(DynamicVector<real>& gamma,
                              DynamicVector<real>& lambda,
                              DynamicVector<real>& f,
                              const uint size);
  void updateNewtonStepVector(DynamicVector<real>& gamma,
                              DynamicVector<real>& lambda,
                              DynamicVector<real>& f,
                              real t,
                              const uint size);
  void conjugateGradient(DynamicVector<real>& x);
  int preconditionedConjugateGradient(DynamicVector<real>& x, const uint size);
  void buildPreconditioner(const uint size);
  void applyPreconditioning(DynamicVector<real>& src, DynamicVector<real>& dst);
  void MultiplyByDiagMatrix(DynamicVector<real>& diagVec, DynamicVector<real>& src, DynamicVector<real>& dst);

  // PDIP specific vectors
  DynamicVector<real> gamma, f, r, lambda, r_d, r_g, ones, delta_gamma, delta_lambda, lambda_tmp, gamma_tmp;
  DynamicVector<real> r_cg, p_cg, z_cg, Ap_cg, prec_cg;
  CompressedMatrix<real> grad_f, M_hat, B, diaglambda, Dinv;
};
}

#endif
