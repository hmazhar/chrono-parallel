#include "chrono_parallel/lcp/ChLcpSolverParallel.h"
#include "chrono_parallel/math/ChThrustLinearAlgebra.h"

#include "chrono_parallel/solver/ChSolverAPGD.h"
#include "chrono_parallel/solver/ChSolverAPGDREF.h"
#include "chrono_parallel/solver/ChSolverBiCG.h"
#include "chrono_parallel/solver/ChSolverBiCGStab.h"
#include "chrono_parallel/solver/ChSolverCG.h"
#include "chrono_parallel/solver/ChSolverCGS.h"
#include "chrono_parallel/solver/ChSolverMinRes.h"
#include "chrono_parallel/solver/ChSolverSD.h"
#include "chrono_parallel/solver/ChSolverGD.h"
#include "chrono_parallel/solver/ChSolverPGS.h"
#include "chrono_parallel/solver/ChSolverJacobi.h"
#include "chrono_parallel/solver/ChSolverPDIP.h"
using namespace chrono;

void ChLcpSolverParallelDVI::RunTimeStep()
{
  // Compute the offsets and number of constrains depending on the solver mode
  if (data_container->settings.solver.solver_mode == NORMAL) {
    rigid_rigid.offset = 1;
    data_container->num_unilaterals = 1 * data_container->num_contacts;
  } else if (data_container->settings.solver.solver_mode == SLIDING) {
    rigid_rigid.offset = 3;
    data_container->num_unilaterals = 3 * data_container->num_contacts;
  } else if (data_container->settings.solver.solver_mode == SPINNING) {
    rigid_rigid.offset = 6;
    data_container->num_unilaterals = 6 * data_container->num_contacts;
  }
  // This is the total number of constraints
  data_container->num_constraints = data_container->num_unilaterals + data_container->num_bilaterals;

  // Generate the mass matrix and compute M_inv_k
  ComputeMassMatrix();

  data_container->host_data.gamma.resize(data_container->num_constraints);
  data_container->host_data.gamma.setZero();

  // Perform any setup tasks for all constraint types
  rigid_rigid.Setup(data_container);
  bilateral.Setup(data_container);
  // Clear and reset solver history data and counters
  solver->current_iteration = 0;
  data_container->measures.solver.total_iteration = 0;
  data_container->measures.solver.maxd_hist.clear();
  data_container->measures.solver.maxdeltalambda_hist.clear();
  data_container->measures.solver.iter_hist.clear();
  // Set pointers to constraint objects and perform setup actions for solver
  solver->rigid_rigid = &rigid_rigid;
  solver->bilateral = &bilateral;
  solver->Setup(data_container);

  ComputeD();
  ComputeE();
  ComputeR();

  PreSolve();

  data_container->system_timer.start("ChLcpSolverParallel_Solve");

//  if (data_container->settings.solver.max_iteration_bilateral > 0) {
//    solver->SetMaxIterations(data_container->settings.solver.max_iteration_bilateral);
//    data_container->settings.solver.local_solver_mode = BILATERAL;
//    SetR();
//    solver->Solve();
//  }

  PerformStabilization();

  if (data_container->settings.solver.solver_mode == NORMAL || data_container->settings.solver.solver_mode == SLIDING || data_container->settings.solver.solver_mode == SPINNING) {
    if (data_container->settings.solver.max_iteration_normal > 0) {
      solver->SetMaxIterations(data_container->settings.solver.max_iteration_normal);
      data_container->settings.solver.local_solver_mode = NORMAL;
      SetR();
      solver->Solve();
    }
  }
  if (data_container->settings.solver.solver_mode == SLIDING || data_container->settings.solver.solver_mode == SPINNING) {
    if (data_container->settings.solver.max_iteration_sliding > 0) {
      solver->SetMaxIterations(data_container->settings.solver.max_iteration_sliding);
      data_container->settings.solver.local_solver_mode = SLIDING;
      SetR();
      solver->Solve();
    }
  }
  if (data_container->settings.solver.solver_mode == SPINNING) {
    if (data_container->settings.solver.max_iteration_spinning > 0) {
      solver->SetMaxIterations(data_container->settings.solver.max_iteration_spinning);
      data_container->settings.solver.local_solver_mode = SPINNING;
      SetR();
      solver->Solve();
    }
  }

  data_container->Fc_current = false;
  data_container->system_timer.stop("ChLcpSolverParallel_Solve");
  
  ComputeImpulses();

  for (int i = 0; i < data_container->measures.solver.iter_hist.size(); i++) {
    AtIterationEnd(data_container->measures.solver.maxd_hist[i], data_container->measures.solver.maxdeltalambda_hist[i], data_container->measures.solver.iter_hist[i]);
  }
  tot_iterations = data_container->measures.solver.iter_hist.size();

#if PRINT_LEVEL == 2
  std::cout << "Solve Done: " << residual << std::endl;
#endif
}

void ChLcpSolverParallelDVI::ComputeD()
{
  uint num_constraints = data_container->num_constraints;
  if (num_constraints <= 0) {
    return;
  }

  uint num_bodies = data_container->num_bodies;
  uint num_shafts = data_container->num_shafts;
  uint num_dof = data_container->num_dof;
  uint num_contacts = data_container->num_contacts;
  uint num_bilaterals = data_container->num_bilaterals;
  uint nnz_bilaterals = data_container->nnz_bilaterals;

  int nnz_normal = 6 * 2 * data_container->num_contacts;
  int nnz_tangential = 6 * 4 * data_container->num_contacts;
  int nnz_spinning = 6 * 3 * data_container->num_contacts;

  int num_normal = 1 * data_container->num_contacts;
  int num_tangential = 2 * data_container->num_contacts;
  int num_spinning = 3 * data_container->num_contacts;

  SparseMatrix& D_n_T = data_container->host_data.D_n_T;
  SparseMatrix& D_t_T = data_container->host_data.D_t_T;
  SparseMatrix& D_s_T = data_container->host_data.D_s_T;
  SparseMatrix& D_b_T = data_container->host_data.D_b_T;

  SparseMatrix& D_n = data_container->host_data.D_n;
  SparseMatrix& D_t = data_container->host_data.D_t;
  SparseMatrix& D_s = data_container->host_data.D_s;
  SparseMatrix& D_b = data_container->host_data.D_b;

  SparseMatrix& M_invD_n = data_container->host_data.M_invD_n;
  SparseMatrix& M_invD_t = data_container->host_data.M_invD_t;
  SparseMatrix& M_invD_s = data_container->host_data.M_invD_s;
  SparseMatrix& M_invD_b = data_container->host_data.M_invD_b;

  const SparseMatrix& M_inv = data_container->host_data.M_inv;

  D_n_T.makeCompressed();
  D_t_T.makeCompressed();
  D_s_T.makeCompressed();
  D_b_T.makeCompressed();

  switch (data_container->settings.solver.solver_mode) {
    case NORMAL:
      D_n_T.setZero();

      D_n_T.reserve(nnz_normal);

      D_n_T.resize(num_normal, num_dof);
      break;
    case SLIDING:
      D_n_T.setZero();
      D_t_T.setZero();

      D_n_T.reserve(nnz_normal);
      D_t_T.reserve(nnz_tangential);

      D_n_T.resize(num_normal, num_dof);
      D_t_T.resize(num_tangential, num_dof);
      break;
    case SPINNING:
      D_n_T.setZero();
      D_t_T.setZero();
      D_s_T.setZero();

      D_n_T.reserve(nnz_normal);
      D_t_T.reserve(nnz_tangential);
      D_s_T.reserve(nnz_spinning);

      D_n_T.resize(num_normal, num_dof );
      D_t_T.resize(num_tangential, num_dof);
      D_s_T.resize(num_spinning, num_dof);
      break;
  }



  D_b_T.setZero();
  D_b_T.reserve(nnz_bilaterals);
  D_b_T.resize(num_bilaterals, num_dof);

  rigid_rigid.GenerateSparsity();
  bilateral.GenerateSparsity();
  rigid_rigid.Build_D();
  bilateral.Build_D();

  switch (data_container->settings.solver.solver_mode) {
    case NORMAL:
      D_n = D_n_T.transpose();
      M_invD_n = M_inv * D_n;
      break;
    case SLIDING:
      D_n = D_n_T.transpose();
      D_t = D_t_T.transpose();
      M_invD_n = M_inv * D_n;
      M_invD_t = M_inv * D_t;
      break;
    case SPINNING:
      D_n = D_n_T.transpose();
      D_t = D_t_T.transpose();
      D_s = D_s_T.transpose();
      M_invD_n = M_inv * D_n;
      M_invD_t = M_inv * D_t;
      M_invD_s = M_inv * D_s;
      break;
  }

  D_b = D_b_T.transpose();
  M_invD_b = M_inv * D_b;
}

void ChLcpSolverParallelDVI::ComputeE()
{
  if (data_container->num_constraints <= 0) {
    return;
  }

  data_container->host_data.E.resize(data_container->num_constraints);

  rigid_rigid.Build_E();
  bilateral.Build_E();
}

void ChLcpSolverParallelDVI::ComputeR() {
  if (data_container->num_constraints <= 0) {
    return;
  }

  const SparseMatrix& D_n_T = data_container->host_data.D_n_T;
  const SparseMatrix& D_t_T = data_container->host_data.D_t_T;
  const SparseMatrix& D_s_T = data_container->host_data.D_s_T;
  const SparseMatrix& D_b_T = data_container->host_data.D_b_T;

  const DenseVector& M_invk = data_container->host_data.M_invk;

  DenseVector& R = data_container->host_data.R_full;
  DenseVector& b = data_container->host_data.b;

  uint num_contacts = data_container->num_contacts;
  uint num_unilaterals = data_container->num_unilaterals;
  uint num_bilaterals = data_container->num_bilaterals;

  b.resize(data_container->num_constraints);
  R.resize(data_container->num_constraints);

  rigid_rigid.Build_b();
  bilateral.Build_b();

  Eigen::VectorBlock<DenseVector> b_n = b.segment(0, num_contacts);
  Eigen::VectorBlock<DenseVector> R_n = R.segment( 0, num_contacts);

  Eigen::VectorBlock<DenseVector> b_b = b.segment( num_unilaterals, num_bilaterals);
  Eigen::VectorBlock<DenseVector> R_b = R.segment( num_unilaterals, num_bilaterals);

  R_b = -b_b - D_b_T * M_invk;
  switch (data_container->settings.solver.solver_mode) {
    case NORMAL: {
      R_n = -b_n - D_n_T * M_invk;
    } break;

    case SLIDING: {

      //blaze::DenseSubvector<DenseVector > b_t = blaze::subvector(b, num_contacts, num_contacts * 2);
      Eigen::VectorBlock<DenseVector> R_t = R.segment( num_contacts, num_contacts * 2);

      R_n = -b_n - D_n_T * M_invk;
      R_t = - D_t_T * M_invk;
    } break;

    case SPINNING: {
      //blaze::DenseSubvector<DenseVector > b_t = blaze::subvector(b, num_contacts, num_contacts * 2);
      Eigen::VectorBlock<DenseVector> R_t = R.segment( num_contacts, num_contacts * 2);

      //blaze::DenseSubvector<DenseVector > b_s = blaze::subvector(b, num_contacts * 3, num_contacts * 3);
      Eigen::VectorBlock<DenseVector> R_s = R.segment( num_contacts * 3, num_contacts * 3);

      R_n = -b_n - D_n_T * M_invk;
      R_t =  - D_t_T * M_invk;
      R_s =  - D_s_T * M_invk;
    } break;
  }
}

void ChLcpSolverParallelDVI::SetR() {
  if (data_container->num_constraints <= 0) {
    return;
  }

  DenseVector& R = data_container->host_data.R;
  const DenseVector& R_full = data_container->host_data.R_full;

  uint num_contacts = data_container->num_contacts;
  uint num_unilaterals = data_container->num_unilaterals;
  uint num_bilaterals = data_container->num_bilaterals;
  R.resize(data_container->num_constraints);
  R.setZero();

  switch (data_container->settings.solver.local_solver_mode) {
    case BILATERAL:{
      R.segment(num_unilaterals, num_bilaterals) = R_full.segment(num_unilaterals, num_bilaterals);
    } break;

    case NORMAL: {
      R.segment( num_unilaterals, num_bilaterals) = R_full.segment( num_unilaterals, num_bilaterals);
      R.segment( 0, num_contacts) = R_full.segment( 0, num_contacts);
    } break;
    case SLIDING: {
      R.segment( num_unilaterals, num_bilaterals) = R_full.segment( num_unilaterals, num_bilaterals);
      R.segment( 0, num_contacts) = R_full.segment( 0, num_contacts);
      R.segment( num_contacts, num_contacts * 2) =R_full.segment( num_contacts, num_contacts * 2);
    } break;

    case SPINNING: {
      R.segment( num_unilaterals, num_bilaterals) = R_full.segment( num_unilaterals, num_bilaterals);
      R.segment( 0, num_contacts) = R_full.segment( 0, num_contacts);
      R.segment( num_contacts, num_contacts * 2) =R_full.segment(num_contacts, num_contacts * 2);
      R.segment( num_contacts * 3, num_contacts * 3) = R_full.segment( num_contacts * 3, num_contacts * 3);
    } break;
  }
}




void ChLcpSolverParallelDVI::ComputeImpulses() {

  DenseVector& v = data_container->host_data.v;

  const DenseVector& M_invk = data_container->host_data.M_invk;
  const DenseVector& gamma = data_container->host_data.gamma;

  const SparseMatrix& M_invD_n = data_container->host_data.M_invD_n;
  const SparseMatrix& M_invD_t = data_container->host_data.M_invD_t;
  const SparseMatrix& M_invD_s = data_container->host_data.M_invD_s;
  const SparseMatrix& M_invD_b = data_container->host_data.M_invD_b;

  uint num_contacts = data_container->num_contacts;
  uint num_unilaterals = data_container->num_unilaterals;
  uint num_bilaterals = data_container->num_bilaterals;

  if (data_container->num_constraints > 0) {

    Eigen::VectorBlock<const DenseVector> gamma_b = gamma.segment( num_unilaterals, num_bilaterals);
    Eigen::VectorBlock<const DenseVector> gamma_n = gamma.segment( 0, num_contacts);

    //Compute new velocity based on the lagrange multipliers
    switch (data_container->settings.solver.solver_mode) {
      case NORMAL: {
        v = M_invk + M_invD_n * gamma_n + M_invD_b * gamma_b;
      } break;

      case SLIDING: {
        Eigen::VectorBlock<const DenseVector> gamma_t = gamma.segment( num_contacts, num_contacts * 2);

        v = M_invk + M_invD_n * gamma_n + M_invD_t * gamma_t + M_invD_b * gamma_b;
        //printf("-gamma: %f %f %f \n", gamma_n[0],gamma_t[0],gamma_t[1]);
      } break;

      case SPINNING: {
        Eigen::VectorBlock<const DenseVector> gamma_t = gamma.segment( num_contacts, num_contacts * 2);
        Eigen::VectorBlock<const DenseVector> gamma_s = gamma.segment( num_contacts * 3, num_contacts * 3);

        v = M_invk + M_invD_n * gamma_n + M_invD_t * gamma_t + M_invD_s * gamma_s + M_invD_b * gamma_b;

      } break;
    }
  } else {
    //When there are no constraints we need to still apply gravity and other
    //body forces!
    v = M_invk;
  }
}

void ChLcpSolverParallelDVI::PreSolve(){
  if (data_container->num_contacts <= 0) {
    return;
  }

  if (data_container->settings.solver.solver_mode == NORMAL) {
    return;
  }

  if(data_container->settings.solver.presolve){
    //rigid_rigid.SolveLocal();
  }

}



void ChLcpSolverParallelDVI::ChangeSolverType(SOLVERTYPE type) {
  data_container->settings.solver.solver_type = type;

  if (this->solver) {
    delete (this->solver);
  }
  switch (type) {
    case STEEPEST_DESCENT:
      solver = new ChSolverSD();
      break;
    case GRADIENT_DESCENT:
      solver = new ChSolverGD();
      break;
    case CONJUGATE_GRADIENT:
      solver = new ChSolverCG();
      break;
    case CONJUGATE_GRADIENT_SQUARED:
      solver = new ChSolverCGS();
      break;
    case BICONJUGATE_GRADIENT:
      solver = new ChSolverBiCG();
      break;
    case BICONJUGATE_GRADIENT_STAB:
      solver = new ChSolverBiCGStab();
      break;
    case MINIMUM_RESIDUAL:
      solver = new ChSolverMinRes();
      break;
    case QUASAI_MINIMUM_RESIDUAL:
      // This solver has not been implemented yet
      break;
    case APGD:
      solver = new ChSolverAPGD();
      break;
    case APGDREF:
      solver = new ChSolverAPGDREF();
      break;
    case JACOBI:
      solver = new ChSolverJacobi();
      break;
    case GAUSS_SEIDEL:
      solver = new ChSolverPGS();
      break;
    case PDIP:
      solver = new ChSolverPDIP();
      break;
  }

}
