#include "chrono_parallel/constraints/ChConstraintRigidFluid.h"
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
void Orthog(real3& Vx, real3& Vy, real3& Vz) {
  real3 mVsingular = R3(0, 1, 0);
  Vz = cross(Vx, mVsingular);
  real mzlen = Vz.length();

  if (mzlen < real(0.0001)) {  // was near singularity? change singularity
                               // reference custom_vector!

    mVsingular = R3(1, 0, 0);

    Vz = cross(Vx, mVsingular);
    mzlen = Vz.length();
  }
  Vz = Vz / mzlen;
  Vy = cross(Vz, Vx);
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
    real3 n = norm[index], V, W;

    real4 quaternion_conjugate = ~rot_rigid[rigid];
    real3 sbar = quatRotate(cpta[index], quaternion_conjugate);
    real3 T1 = cross(quatRotate(n, quaternion_conjugate), sbar);

    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 0) = -n.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 1) = -n.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 2) = -n.z;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 3) = T1.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 4) = T1.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 5) = T1.z;

    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0) = n.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1) = n.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2) = n.z;

    Orthog(n, V, W);
    real3 T2 = cross(quatRotate(V, quaternion_conjugate), sbar);
    real3 T3 = cross(quatRotate(W, quaternion_conjugate), sbar);

    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 0) = -V.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 1) = -V.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 2) = -V.z;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 3) = T2.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 4) = T2.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 5) = T2.z;

    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0) = V.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1) = V.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2) = V.z;

    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 0) = -W.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 1) = -W.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 2) = -W.z;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 3) = T3.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 4) = T3.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 5) = T3.z;

    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0) = W.x;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1) = W.y;
    D_T(num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2) = W.z;
  }
  data_manager->system_timer.stop("ChSolverParallel_solverC");
}
void ChConstraintRigidFluid::Build_b() {
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

    bi = std::max(real(1.0) / step_size * depth, -data_manager->settings.solver.contact_recovery_speed);
    //
    data_manager->host_data.b[num_unilaterals + num_bilaterals + index * 3 + 0] = bi;
    data_manager->host_data.b[num_unilaterals + num_bilaterals + index * 3 + 1] = 0;
    data_manager->host_data.b[num_unilaterals + num_bilaterals + index * 3 + 2] = 0;
  }
}
void ChConstraintRigidFluid::Build_E() {
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
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 0, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 1, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 2, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 3, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 4, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, rigid * 6 + 5, 0);

    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 0, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2, 0);
    D_T.finalize(num_unilaterals + num_bilaterals + index * 3 + 0);

    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 0, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 1, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 2, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 3, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 4, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, rigid * 6 + 5, 0);

    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 1, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2, 0);
    D_T.finalize(num_unilaterals + num_bilaterals + index * 3 + 1);

    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 0, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 1, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 2, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 3, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 4, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, rigid * 6 + 5, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 0, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 1, 0);
    D_T.append(num_unilaterals + num_bilaterals + index * 3 + 2, num_rigid_bodies * 6 + num_shafts + fluid * 3 + 2, 0);
    D_T.finalize(num_unilaterals + num_bilaterals + index * 3 + 2);
  }
}
