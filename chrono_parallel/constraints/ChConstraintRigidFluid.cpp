#include "chrono_parallel/constraints/ChConstraintRigidFluid.h"
#include "chrono_parallel/constraints/ChConstraintUtils.h"

using namespace chrono;

bool Cone_generalized_rf(real& gamma_n, real& gamma_u, real& gamma_v, const real& mu) {
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

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void ChConstraintRigidFluid::Project(real* gamma) {
  custom_vector<int2>& bids = data_manager->host_data.bids_rigid_fluid;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
#pragma omp parallel for
  for (int index = 0; index < num_rigid_fluid_contacts; index++) {
    int2 body_id = bids[index];
    int rigid = body_id.x;  // rigid is stored in the first index
    int fluid = body_id.y;  // fluid body is in second index
    real cohesion =
        std::max((data_manager->host_data.cohesion_data[rigid] + data_manager->settings.fluid.cohesion) * .5, 0.0);
    real friction = (data_manager->host_data.fric_data[rigid].x == 0 || data_manager->settings.fluid.mu == 0)
                        ? 0
                        : (data_manager->host_data.fric_data[rigid].x + data_manager->settings.fluid.mu) * .5;
    //      gamma[num_unilaterals + num_bilaterals + index] =
    //      std::max(real(0.0), gamma[num_unilaterals + num_bilaterals +
    //      index]);

    // real gam = gamma[num_unilaterals + num_bilaterals + index];
    //    gam = gam + cohesion;
    //    gam = gam < 0 ? 0 : gam - cohesion;
    //    gamma[num_unilaterals + num_bilaterals + index] = gam;

    //      int _index_ = num_unilaterals + num_bilaterals + index*3;
    //
    //
    //
    real3 gam;
    gam.x = gamma[num_unilaterals + num_bilaterals + index * 3 + 0];
    gam.y = gamma[num_unilaterals + num_bilaterals + index * 3 + 1];
    gam.z = gamma[num_unilaterals + num_bilaterals + index * 3 + 2];

    gam.x += cohesion;

    real mu = friction;
    if (mu == 0) {
      gam.x = gam.x < 0 ? 0 : gam.x - cohesion;
      gam.y = gam.z = 0;

      gamma[num_unilaterals + num_bilaterals + index * 3 + 0] = gam.x;
      gamma[num_unilaterals + num_bilaterals + index * 3 + 1] = gam.y;
      gamma[num_unilaterals + num_bilaterals + index * 3 + 2] = gam.z;

      continue;
    }

    if (Cone_generalized_rf(gam.x, gam.y, gam.z, mu)) {
    }

    gamma[num_unilaterals + num_bilaterals + index * 3 + 0] = gam.x - cohesion;
    gamma[num_unilaterals + num_bilaterals + index * 3 + 1] = gam.y;
    gamma[num_unilaterals + num_bilaterals + index * 3 + 2] = gam.z;
  }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void ChConstraintRigidFluid::Build_D() {
  LOG(INFO) << "ChConstraintRigidFluid::Build_D";

  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_rigid_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;

  if (num_rigid_fluid_contacts <= 0) {
    return;
  }
  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
  custom_vector<real3>& pos = data_manager->host_data.pos_fluid;

  custom_vector<real3>& pos_rigid = data_manager->host_data.pos_rigid;
  custom_vector<real4>& rot_rigid = data_manager->host_data.rot_rigid;

  real h = data_manager->settings.fluid.kernel_radius;
  custom_vector<int2>& bids = data_manager->host_data.bids_rigid_fluid;
  custom_vector<real3>& cpta = data_manager->host_data.cpta_rigid_fluid;
  custom_vector<real3>& norm = data_manager->host_data.norm_rigid_fluid;
  data_manager->system_timer.start("ChSolverParallel_solverC");

#pragma omp paralle for
  for (int index = 0; index < num_rigid_fluid_contacts; index++) {
    int2 body_id = bids[index];
    int rigid = body_id.x;  // rigid is stored in the first index
    int fluid = body_id.y;  // fluid body is in second index
    real3 U = norm[index], V, W;
    Orthogonalize(U, V, W);
    real3 T1, T2, T3;
    Compute_Jacobian(rot_rigid[rigid], U, V, W, cpta[index] - pos_rigid[rigid], T1, T2, T3);

    SetRow6(D_T, num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6, -U, T1);
    SetRow6(D_T, num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6, -V, T2);
    SetRow6(D_T, num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6, -W, T3);

    SetRow3(D_T, num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3, U);
    SetRow3(D_T, num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3, V);
    SetRow3(D_T, num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3, W);
  }
  data_manager->system_timer.stop("ChSolverParallel_solverC");
}
void ChConstraintRigidFluid::Build_b() {
  LOG(INFO) << "ChConstraintRigidFluid::Build_b";
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  real step_size = data_manager->settings.step_size;
  if (num_rigid_fluid_contacts <= 0) {
    return;
  }
#pragma omp parallel for
  for (int index = 0; index < num_rigid_fluid_contacts; index++) {
    real bi = 0;
    real depth = data_manager->host_data.dpth_rigid_fluid[index];

    bi = std::max(real(1.0) / step_size * depth, -data_manager->settings.fluid.contact_recovery_speed);
    //
    data_manager->host_data.b[num_unilaterals + num_bilaterals + index * 3 + 0] = bi;
    data_manager->host_data.b[num_unilaterals + num_bilaterals + index * 3 + 1] = 0;
    data_manager->host_data.b[num_unilaterals + num_bilaterals + index * 3 + 2] = 0;
  }
}
void ChConstraintRigidFluid::Build_E() {
  LOG(INFO) << "ChConstraintRigidFluid::Build_E";
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  real step_size = data_manager->settings.step_size;
  if (num_rigid_fluid_contacts <= 0) {
    return;
  }
  DynamicVector<real>& E = data_manager->host_data.E;

#pragma omp parallel for
  for (int index = 0; index < num_rigid_fluid_contacts; index++) {
    int ind = num_unilaterals + num_bilaterals;
    real epsilon = data_manager->settings.fluid.epsilon;
    real tau = data_manager->settings.fluid.tau;
    real h = data_manager->settings.fluid.kernel_radius;
    real compliance = 4.0 / (step_size * step_size) * (epsilon / (1.0 + 4.0 * tau / h));

    E[ind + index * 3 + 0] = 0;
    E[ind + index * 3 + 1] = 0;
    E[ind + index * 3 + 2] = 0;
  }
}

void ChConstraintRigidFluid::GenerateSparsity() {
  LOG(INFO) << "ChConstraintRigidFluid::GenerateSparsity";
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  real step_size = data_manager->settings.step_size;
  uint num_rigid_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;
  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
  custom_vector<int2>& bids = data_manager->host_data.bids_rigid_fluid;
  for (int index = 0; index < num_rigid_fluid_contacts; index++) {
    int2 body_id = bids[index];
    int rigid = body_id.x;
    int fluid = body_id.y;
    int off = num_unilaterals + num_bilaterals;

    D_T.append(off + index * 3 + 0, rigid * 6 + 0, 1);
    D_T.append(off + index * 3 + 0, rigid * 6 + 1, 1);
    D_T.append(off + index * 3 + 0, rigid * 6 + 2, 1);
    D_T.append(off + index * 3 + 0, rigid * 6 + 3, 1);
    D_T.append(off + index * 3 + 0, rigid * 6 + 4, 1);
    D_T.append(off + index * 3 + 0, rigid * 6 + 5, 1);
    D_T.append(off + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0, 1);
    D_T.append(off + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1, 1);
    D_T.append(off + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2, 1);
    D_T.finalize(off + index * 3 + 0);

    D_T.append(off + index * 3 + 1, rigid * 6 + 0, 1);
    D_T.append(off + index * 3 + 1, rigid * 6 + 1, 1);
    D_T.append(off + index * 3 + 1, rigid * 6 + 2, 1);
    D_T.append(off + index * 3 + 1, rigid * 6 + 3, 1);
    D_T.append(off + index * 3 + 1, rigid * 6 + 4, 1);
    D_T.append(off + index * 3 + 1, rigid * 6 + 5, 1);
    D_T.append(off + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0, 1);
    D_T.append(off + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1, 1);
    D_T.append(off + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2, 1);
    D_T.finalize(off + index * 3 + 1);

    D_T.append(off + index * 3 + 2, rigid * 6 + 0, 1);
    D_T.append(off + index * 3 + 2, rigid * 6 + 1, 1);
    D_T.append(off + index * 3 + 2, rigid * 6 + 2, 1);
    D_T.append(off + index * 3 + 2, rigid * 6 + 3, 1);
    D_T.append(off + index * 3 + 2, rigid * 6 + 4, 1);
    D_T.append(off + index * 3 + 2, rigid * 6 + 5, 1);
    D_T.append(off + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0, 1);
    D_T.append(off + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1, 1);
    D_T.append(off + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2, 1);
    D_T.finalize(off + index * 3 + 2);
  }
}
