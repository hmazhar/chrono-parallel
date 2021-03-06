#include "chrono_parallel/solver/ChSolverAPGD.h"
#include <blaze/math/CompressedVector.h>
using namespace chrono;

ChSolverAPGD::ChSolverAPGD()
    : ChSolverParallel(),
      mg_tmp_norm(0),
      mb_tmp_norm(0),
      obj1(0),
      obj2(0),
      norm_ms(0),
      dot_g_temp(0),
      theta(1),
      theta_new(0),
      beta_new(0),
      t(1),
      L(1),
      g_diff(0) {
}

void ChSolverAPGD::UpdateR() {
  const SubMatrixType& D_n_T = _DNT_;
  const DynamicVector<real>& M_invk = data_manager->host_data.M_invk;
  const DynamicVector<real>& b = data_manager->host_data.b;
  DynamicVector<real>& R = data_manager->host_data.R;
  DynamicVector<real>& s = data_manager->host_data.s;

  uint num_contacts = data_manager->num_rigid_contacts;

  s.resize(data_manager->num_rigid_contacts);
  reset(s);

  rigid_rigid->Build_s();

  ConstSubVectorType b_n = blaze::subvector(b, 0, num_contacts);
  SubVectorType R_n = blaze::subvector(R, 0, num_contacts);
  SubVectorType s_n = blaze::subvector(s, 0, num_contacts);

  R_n = -b_n - D_n_T * M_invk + s_n;
}

uint ChSolverAPGD::SolveAPGD(const uint max_iter,
                             const uint size,
                             const DynamicVector<real>& r,
                             DynamicVector<real>& gamma) {
  real& residual = data_manager->measures.solver.residual;
  real& objective_value = data_manager->measures.solver.objective_value;
  real& old_objective_value = data_manager->measures.solver.old_objective_value;
  old_objective_value = LARGE_REAL;
  // LOG(TRACE) << "APGD START";
  DynamicVector<real> one(size, 1.0);

  gamma_hat.resize(size);
  N_gamma_new.resize(size);
  temp.resize(size);
  g.resize(size);
  gamma_new.resize(size);
  y.resize(size);

  residual = 10e30;
  g_diff = 1.0 / pow(size, 2.0);

  theta = 1;
  theta_new = theta;
  beta_new = 0.0;
  mb_tmp_norm = 0, mg_tmp_norm = 0;
  obj1 = 0.0, obj2 = 0.0;
  dot_g_temp = 0, norm_ms = 0;

  // Is the initial projection necessary?
  // Project(gamma.data());
  // gamma_hat = gamma;
  // ShurProduct(gamma, mg);
  // mg = mg - r;
  data_manager->system_timer.start("ChSolverParallel_Solve");
  temp = gamma - one;
  real norm_temp = sqrt((real)(temp, temp));
  if (data_manager->settings.solver.apgd_adaptive_step) {
    if (data_manager->settings.solver.cache_step_length == false) {
      if (norm_temp == 0) {  // If gamma is one temp should be zero, in that case set L to one. We cannot divide by 0
        L = 5.0;
      } else {  // If the N matrix is zero for some reason, temp will be zero
        ShurProduct(temp, temp);
        L = sqrt((real)(temp, temp)) / norm_temp;  // If temp is zero then L will be zero
      }
      // When L is zero the step length can't be computed, in this case just return
      // If the N is indeed zero then solving doesn't make sense
      if (L == 0) {
        // For certain simulations returning here will not perform any iterations even when there are contacts that
        // aren't resolved. Changed it from return 0 to L=t=1;
        // return 0;
        L = 5.0;
        t = 1.0/L;
      } else {
        t = 1.0 / L;  // Compute the step size
      }
    } else {
    }
  } else {
    L = data_manager->settings.solver.apgd_step_size;
    t = 1.0 / L;
  }
  y = gamma;
  // If no iterations are performed or the residual is NAN (which is shouldn't be) make sure that gamma_hat has
  // something inside of it. Otherwise gamma will be overwritten with a vector of zero size
  gamma_hat = gamma;
  data_manager->system_timer.stop("ChSolverParallel_Solve");
  for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {
    data_manager->system_timer.start("ChSolverParallel_Solve");
    ShurProduct(y, g);
    g = g - r;
    gamma_new = y - t * g;

    Project(gamma_new.data());

    ShurProduct(gamma_new, N_gamma_new);
    obj1 = 0.5 * (gamma_new, N_gamma_new) - (gamma_new, r);

    ShurProduct(y, temp);
    obj2 = 0.5 * (y, temp) - (y, r);

    temp = gamma_new - y;
    dot_g_temp = (g, temp);
    norm_ms = (temp, temp);
    real lip = obj2 + dot_g_temp + 0.5 * L * norm_ms;

    if (data_manager->settings.solver.apgd_adaptive_step) {
      while (obj1 > lip) {
        L = 2.0 * L;
        t = 1.0 / L;

        gamma_new = y - t * g;
        Project(gamma_new.data());
        ShurProduct(gamma_new, N_gamma_new);
        obj1 = 0.5 * (gamma_new, N_gamma_new) - (gamma_new, r);
        temp = gamma_new - y;
        dot_g_temp = (g, temp);
        norm_ms = (temp, temp);
        lip = obj2 + dot_g_temp + 0.5 * L * norm_ms;
      }
    } else {
      L = data_manager->settings.solver.apgd_step_size;
      t = 1.0 / L;
    }
    real theta_sqr = pow(theta, 2.0);
    theta_new = (-theta_sqr + theta * sqrt(theta_sqr + 4.0)) * 0.5;
    beta_new = theta * (1.0 - theta) / (theta_sqr + theta_new);
    temp = gamma_new - gamma;
    y = beta_new * temp + gamma_new;
    dot_g_temp = (g, temp);

    if (dot_g_temp > 0) {  // LOG(TRACE) << "APGD RESET " << current_iteration;
      y = gamma_new;
      theta_new = 1.0;
    }
    if (data_manager->settings.solver.apgd_adaptive_step) {
      L = 0.9 * L;
      t = 1.0 / L;
    }
    theta = theta_new;
    gamma = gamma_new;
    data_manager->system_timer.stop("ChSolverParallel_Solve");
    if (current_iteration % data_manager->settings.solver.skip_tolerance_check == 0) {
      // Compute the residual
      temp = gamma_new - g_diff * (N_gamma_new - r);
      // real temp_dota = (real)(temp, temp);
      // ಠ_ಠ THIS PROJECTION IS IMPORTANT! (╯°□°)╯︵ ┻━┻
      // If turned off the residual will be very incorrect! Turning it off can cause the solver to effectively use the
      // solution found in the first step because the residual never get's smaller. (You can convince yourself of this
      // by looking at the objective function value and watch it decrease while the residual and the current solution
      // remain the same.)
      Project(temp.data());
      temp = (1.0 / g_diff) * (gamma_new - temp);
      real res = sqrt((real)(temp, temp));

      if (res < residual) {
        residual = res;
        gamma_hat = gamma_new;
      }
      if (data_manager->settings.solver.test_objective) {
        temp = 0.5 * N_gamma_new - r;
        objective_value = (gamma_new, temp);

        if (fabs(objective_value - old_objective_value) <= data_manager->settings.solver.tolerance_objective) {
          break;
        }
        old_objective_value = objective_value;
      }
      if (residual < data_manager->settings.solver.tol_speed) {
        break;
      }
    }
    AtIterationEnd(residual, objective_value, data_manager->system_timer.GetTime("ChSolverParallel_Solve"));
    data_manager->measures.solver.apgd_beta.push_back(beta_new);
    data_manager->measures.solver.apgd_step.push_back(L);
    temp = N_gamma_new - r;
    real dott = sqrt((real)(temp,temp));
    data_manager->measures.solver.violation.push_back(dott);
    //    if (data_manager->settings.solver.update_rhs) {
    //      UpdateR();
    //    }
  }
  // LOG(TRACE) << "APGD SOLVE COMPLETE";
  data_manager->system_timer.start("ChSolverParallel_Solve");
  gamma = gamma_hat;
  data_manager->system_timer.stop("ChSolverParallel_Solve");
  return current_iteration;
}
