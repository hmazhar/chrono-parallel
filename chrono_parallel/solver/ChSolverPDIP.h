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

class CH_PARALLEL_API ChSolverPDIP: public ChSolverParallel {
public:
  ChSolverPDIP() : ChSolverParallel() {}
  ~ChSolverPDIP() {}

  void Solve() {
    if (data_container->num_constraints == 0) {
      return;
    }
    data_container->system_timer.start("ChSolverParallel_Solve");
    const SparseMatrix& M_inv = data_container->host_data.M_inv;
    uint num_dof = data_container->num_dof;
    uint num_contacts = data_container->num_contacts;
    uint num_bilaterals = data_container->num_bilaterals;
    uint num_constraints = data_container->num_constraints;
    uint num_unilaterals = data_container->num_unilaterals;
    uint nnz_bilaterals = data_container->nnz_bilaterals;
    uint nnz_unilaterals = 6 * 6 * data_container->num_contacts;

    int nnz_total = nnz_unilaterals + nnz_bilaterals;

    D_T.reserve(nnz_total);
    D_T.resize(num_constraints, num_dof);

    D.reserve(nnz_total);
    D.resize(num_dof, num_constraints);

    M_invD.reserve(nnz_total);
    M_invD.resize(num_dof, num_constraints);


    D_T.middleRows(0, num_contacts) = data_container->host_data.D_n_T;
    D_T.middleRows(num_contacts, 2 * num_contacts) = data_container->host_data.D_t_T;
    D_T.middleRows(num_unilaterals, num_bilaterals) = data_container->host_data.D_b_T;


    //Eigen is not able to to set a subset of columns for a rowmajor matrix. Therefore recompute
    D = D_T.transpose();
    M_invD = data_container->host_data.M_inv*D;

    data_container->measures.solver.total_iteration +=
      SolvePDIP(max_iteration,
                data_container->num_constraints,
                data_container->host_data.R,
                data_container->host_data.gamma);
    data_container->system_timer.stop("ChSolverParallel_Solve");
  }

  // Solve using the primal-dual interior point method
  uint SolvePDIP(const uint max_iter,                    // Maximum number of iterations
                 const uint size,                        // Number of unknowns
                 const DenseVector& b,    // Rhs vector
                 DenseVector& x           // The vector of unknowns
                 );

  // Compute the residual for the solver
  real Res4(DenseVector & gamma,
            DenseVector & tmp);

  // Compute the Schur Complement Product, dst = N * src
  void SchurComplementProduct(DenseVector & src, DenseVector & dst);
  void initializeNewtonStepMatrix(DenseVector & gamma, DenseVector & lambda, DenseVector & f, const uint size);
  void initializeConstraintGradient(DenseVector & src, const uint size);
  void getConstraintVector(DenseVector & src, DenseVector & dst, const uint size);
  void updateConstraintGradient(DenseVector & src, const uint size);
  void updateNewtonStepMatrix(DenseVector & gamma, DenseVector & lambda, DenseVector & f, const uint size);
  void updateNewtonStepVector(DenseVector & gamma, DenseVector & lambda, DenseVector & f, real t, const uint size);
  void conjugateGradient(DenseVector & x);
  int preconditionedConjugateGradient(DenseVector & x, const uint size);
  void buildPreconditioner(const uint size);
  void applyPreconditioning(DenseVector & src, DenseVector & dst);
  void MultiplyByDiagMatrix(DenseVector & diagVec, DenseVector & src, DenseVector & dst);

  //PDIP specific vectors
  DenseVector gamma, f, r, lambda, r_d, r_g, ones, delta_gamma, delta_lambda, lambda_tmp, gamma_tmp;
  DenseVector r_cg, p_cg, z_cg, Ap_cg, prec_cg;
  SparseMatrix grad_f, M_hat, B, diaglambda, Dinv;

  SparseMatrix D_T, D, M_invD;

};

}

#endif
