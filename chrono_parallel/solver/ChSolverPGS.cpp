#include "chrono_parallel/solver/ChSolverPGS.h"
#include <blaze/math/SparseRow.h>
#include <blaze/math/CompressedVector.h>
using namespace chrono;

bool cone_gen(real& gamma_n, real& gamma_u, real& gamma_v, const real& mu) {
  real f_tang = sqrt(gamma_u * gamma_u + gamma_v * gamma_v);

  // inside upper cone? keep untouched!
  if (f_tang < (mu * gamma_n)) {
    return false;
  }

  // inside lower cone? reset  normal,u,v to zero!
  if ((f_tang) < -(1.0 / mu) * gamma_n || (fabs(gamma_n) < 10e-15)) {
    gamma_n = 0;
    gamma_u = 0;
    gamma_v = 0;
    return false;
  }

  // remaining case: project orthogonally to generator segment of upper cone
  gamma_n = (f_tang * mu + gamma_n) / (mu * mu + 1);
  real tproj_div_t = (gamma_n * mu) / f_tang;
  gamma_u *= tproj_div_t;
  gamma_v *= tproj_div_t;

  return true;
}

void proj_normal(const real cohesion, real3& gamma, int mode) {
  real coh = cohesion;

  gamma.x += coh;
  gamma.x = gamma.x < 0 ? 0 : gamma.x - coh;
  if (mode == SLIDING) {
    gamma.y = 0;
    gamma.z = 0;
  }
}

void proj_sliding(const real fric, const real cohesion, real3& gamma) {
  real coh = cohesion;
  gamma.x += coh;

  real mu = fric;
  if (mu == 0) {
    gamma.x = gamma.x < 0 ? 0 : gamma.x - coh;
    gamma.y = gamma.z = 0;

    return;
  }

  if (cone_gen(gamma.x, gamma.y, gamma.z, mu)) {
  }

  gamma.x -= coh;
  gamma.y;
  gamma.z;
}

void proj(real friction, real cohesion, real3& gamma, int local_mode, int mode) {
  switch (local_mode) {
    case NORMAL: {
      proj_normal(cohesion, gamma, mode);
    } break;

    case SLIDING: {
      proj_sliding(friction, cohesion, gamma);
    } break;
  }
}

uint ChSolverPGS::SolvePGS(const uint max_iter, const uint size, DynamicVector<real>& mb, DynamicVector<real>& ml) {
  real& residual = data_manager->measures.solver.residual;
  real& objective_value = data_manager->measures.solver.objective_value;
  uint num_contacts = data_manager->num_rigid_contacts;
  CompressedMatrix<real>& Nshur = data_manager->host_data.Nshur;
  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;

  thrust::host_vector<int2>& bids = data_manager->host_data.bids_rigid_rigid;
  thrust::host_vector<real3>& friction = data_manager->host_data.fric_rigid_rigid;
  thrust::host_vector<real>& cohesion = data_manager->host_data.coh_rigid_rigid;

  N_gamma_new.resize(size);
  temp.resize(size);
  ml_new.resize(size);

  real sor = 1;
  diagonal.resize(num_contacts, false);
  real g_diff = 1.0 / pow(size, 2.0);
  real omega = data_manager->settings.solver.omega;

  CompressedMatrix<real> D_T_temp(D_T.rows(), D_T.columns());
  D_T_temp.reserve(D_T.capacity());

  DynamicVector<real> N_gamma_old = Nshur * ml - mb;
  //#pragma omp parallel for
  //    for (size_t i = 0; i < num_contacts; ++i) {
  //        diagonal[i] = 3.0/(
  //        		Nshur(i, i) +
  //				Nshur(num_contacts + i * 2 + 0, num_contacts + i * 2 + 0) +
  //				Nshur(num_contacts + i * 2 + 1,num_contacts + i * 2 + 1)
  //				);
  //      }
  //#pragma omp parallel for
  //  for (int i = 0; i < num_contacts; ++i) {
  //    int a = i * 1 + 0;
  //    int b = num_contacts + i * 2 + 0;
  //    int c = num_contacts + i * 2 + 1;
  //
  //    // real Dinv = 3.0 / (diagonal[a] + diagonal[b] + diagonal[c]);
  //    real Dinv = diagonal[i];
  //    ml[a] = ml[a] - .4 * Dinv * (N_gamma_old[a]);
  //    ml[b] = ml[b] - .4 * Dinv * (N_gamma_old[b]);
  //    ml[c] = ml[c] - .4 * Dinv * (N_gamma_old[c]);
  //  }
  //  Project(ml.data());

  for (size_t i = 0; i < num_contacts; ++i) {
    int a = i * 1 + 0;
    int b = num_contacts + i * 2 + 0;
    int c = num_contacts + i * 2 + 1;

    submatrix(D_T_temp, i * 3 + 0, 0, 1, D_T.columns()) = submatrix(D_T, a, 0, 1, D_T.columns());
    submatrix(D_T_temp, i * 3 + 1, 0, 1, D_T.columns()) = submatrix(D_T, b, 0, 1, D_T.columns());
    submatrix(D_T_temp, i * 3 + 2, 0, 1, D_T.columns()) = submatrix(D_T, c, 0, 1, D_T.columns());

    ml_new[i * 3 + 0] = ml[a];
    ml_new[i * 3 + 1] = ml[b];
    ml_new[i * 3 + 2] = ml[c];
  }
  CompressedMatrix<real> AA = trans(D_T_temp);
  CompressedMatrix<real> N_temp = (D_T_temp * (data_manager->host_data.M_inv * AA));

  data_manager->system_timer.start("ChSolverParallel_Solve");
#pragma omp parallel for
  for (size_t i = 0; i < num_contacts; ++i) {
    diagonal[i] = 3.0 / (N_temp(i * 3 + 0, i * 3 + 0) + N_temp(i * 3 + 1, i * 3 + 1) + N_temp(i * 3 + 2, i * 3 + 2));
  }
  data_manager->system_timer.stop("ChSolverParallel_Solve");

  for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {
    data_manager->system_timer.start("ChSolverParallel_Solve");
    for (size_t i = 0; i < num_contacts; ++i) {
      int a = i * 1 + 0;
      int b = num_contacts + i * 2 + 0;
      int c = num_contacts + i * 2 + 1;
      real Dinv = diagonal[i];

      ml_new[i * 3 + 0] = ml_new[i * 3 + 0] - omega * Dinv * ((row(N_temp, i * 3 + 0), ml_new) - mb[a]);
      ml_new[i * 3 + 1] = ml_new[i * 3 + 1] - omega * Dinv * ((row(N_temp, i * 3 + 1), ml_new) - mb[b]);
      ml_new[i * 3 + 2] = ml_new[i * 3 + 2] - omega * Dinv * ((row(N_temp, i * 3 + 2), ml_new) - mb[c]);
      real3 delta = real3(ml_new[i * 3 + 0], ml_new[i * 3 + 1], ml_new[i * 3 + 2]);

      proj(friction[i].x, cohesion[i], delta, data_manager->settings.solver.local_solver_mode,
           data_manager->settings.solver.solver_mode);

      ml_new[i * 3 + 0] = delta.x;
      ml_new[i * 3 + 1] = delta.y;
      ml_new[i * 3 + 2] = delta.z;
    }
    data_manager->system_timer.stop("ChSolverParallel_Solve");
#pragma omp parallel for
    for (size_t i = 0; i < num_contacts; ++i) {
      int a = i * 1 + 0;
      int b = num_contacts + i * 2 + 0;
      int c = num_contacts + i * 2 + 1;

      ml[a] = ml_new[i * 3 + 0];
      ml[b] = ml_new[i * 3 + 1];
      ml[c] = ml_new[i * 3 + 2];
    }

    ShurProduct(ml, N_gamma_new);
    temp = ml - g_diff * (N_gamma_new - mb);
    Project(temp.data());
    temp = (1.0 / g_diff) * (ml - temp);
    real temp_dotb = (real)(temp, temp);
    real residual = sqrt(temp_dotb);
    objective_value = GetObjective(ml, mb);
    AtIterationEnd(residual, objective_value, data_manager->system_timer.GetTime("ChSolverParallel_Solve"));
  }

  return current_iteration;
  return 0;
}
