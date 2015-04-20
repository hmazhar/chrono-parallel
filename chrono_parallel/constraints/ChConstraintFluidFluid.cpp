

#include "chrono_parallel/constraints/ChConstraintFluidFluid.h"
namespace chrono {

void ChConstraintFluidFluid::Project(real* gamma) {
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_fluid_contacts = data_manager->num_fluid_contacts;
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
  } else {
#pragma omp parallel for
    for (int index = 0; index < num_fluid_contacts; index++) {
      real cohesion = data_manager->settings.fluid.cohesion;
      real gam = gamma[num_unilaterals + num_bilaterals + num_rigid_fluid_contacts * 3 + index];
      gam = gam + cohesion;
      gam = gam < 0 ? 0 : gam - cohesion;
      gamma[num_unilaterals + num_bilaterals + num_rigid_fluid_contacts * 3 + index] = gam;
    }
  }
}

void ChConstraintFluidFluid::Build_D() {
  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
  host_vector<real3>& pos_fluid = data_manager->host_data.pos_fluid;
  uint num_fluid_bodies = data_manager->num_fluid_bodies;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_fluid_contacts = data_manager->num_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_rigid_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;
  real step_size = data_manager->settings.step_size;
  uint index_offset = num_unilaterals + num_bilaterals + num_rigid_fluid_contacts * 3;
  uint body_offset = num_rigid_bodies * 6 + num_shafts;
  custom_vector<int2>& bids = data_manager->host_data.bids_fluid_fluid;

  if (num_fluid_contacts <= 0) {
    return;
  }

#pragma omp paralle for
  for (int index = 0; index < num_fluid_contacts; index++) {
    int2 bid = bids[index];
    real3 n = (pos_fluid[bid.y] - pos_fluid[bid.x]);
    real dist = length(n);
    n = n / dist;
    D_T(index_offset + index, body_offset + bid.x * 3 + 0) = -n.x;
    D_T(index_offset + index, body_offset + bid.x * 3 + 1) = -n.y;
    D_T(index_offset + index, body_offset + bid.x * 3 + 2) = -n.z;

    D_T(index_offset + index, body_offset + bid.y * 3 + 0) = n.x;
    D_T(index_offset + index, body_offset + bid.y * 3 + 1) = n.y;
    D_T(index_offset + index, body_offset + bid.y * 3 + 2) = n.z;
  }
}

void ChConstraintFluidFluid::Build_b() {
  uint num_fluid_bodies = data_manager->num_fluid_bodies;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_fluid_contacts = data_manager->num_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_rigid_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;
  real step_size = data_manager->settings.step_size;
  uint index_offset = num_unilaterals + num_bilaterals + num_rigid_fluid_contacts * 3;
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
  } else {
    blaze::DenseSubvector<DynamicVector<real> > b_sub =
        blaze::subvector(data_manager->host_data.b, index_offset, num_fluid_contacts);
    custom_vector<real3>& pos = data_manager->host_data.pos_fluid;
    custom_vector<int2>& bids = data_manager->host_data.bids_fluid_fluid;
    real h = data_manager->settings.fluid.kernel_radius;
    real inv_hpa = 1.0 / (step_size + data_manager->settings.solver.alpha);
    real inv_hhpa = 1.0 / (step_size * (step_size + data_manager->settings.solver.alpha));
#pragma omp parallel for
    for (int index = 0; index < num_fluid_contacts; index++) {
      int2 bid = bids[index];
      real3 n = (pos[bid.y] - pos[bid.x]);
      real depth = length(n) - h * 2;
      if (depth < 0) {
        real bi = 0;
        if (data_manager->settings.solver.alpha > 0) {
          bi = inv_hpa * depth;
        } else {
          if (data_manager->settings.solver.contact_recovery_speed < 0) {
            bi = real(1.0) / step_size * depth;
          } else {
            bi = std::max(real(1.0) / step_size * depth, -data_manager->settings.solver.contact_recovery_speed);
          }
        }
        b_sub[index] = bi;
      }
    }
  }
}
void ChConstraintFluidFluid::Build_E() {
  DynamicVector<real>& E = data_manager->host_data.E;
  uint num_fluid_bodies = data_manager->num_fluid_bodies;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_fluid_contacts = data_manager->num_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_rigid_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;

  real step_size = data_manager->settings.step_size;
  uint index_offset = num_unilaterals + num_bilaterals + num_rigid_fluid_contacts * 3;

  if (data_manager->settings.fluid.fluid_is_rigid == false) {
  } else {
    if (num_fluid_contacts <= 0) {
      return;
    }
    real epsilon = data_manager->settings.fluid.epsilon;
    real inv_hhpa = 1.0 / (step_size * (step_size + data_manager->settings.solver.alpha));
#pragma omp parallel for
    for (int index = 0; index < num_fluid_contacts; index++) {
      E[index_offset + index] = -inv_hhpa * epsilon;
    }
  }
}

void ChConstraintFluidFluid::GenerateSparsity() {
  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_fluid_contacts = data_manager->num_fluid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_rigid_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;
  if (num_fluid_contacts <= 0) {
    return;
  }
  uint index_offset = num_unilaterals + num_bilaterals + num_rigid_fluid_contacts * 3;
  uint body_offset = num_rigid_bodies * 6 + num_shafts;
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
  } else {
    custom_vector<int2>& bids = data_manager->host_data.bids_fluid_fluid;
    for (int index = 0; index < num_fluid_contacts; index++) {
      int2 bid = bids[index];

      D_T.append(index_offset + index, body_offset + bid.x * 3 + 0, 1);
      D_T.append(index_offset + index, body_offset + bid.x * 3 + 1, 1);
      D_T.append(index_offset + index, body_offset + bid.x * 3 + 2, 1);

      D_T.append(index_offset + index, body_offset + bid.y * 3 + 0, 1);
      D_T.append(index_offset + index, body_offset + bid.y * 3 + 1, 1);
      D_T.append(index_offset + index, body_offset + bid.y * 3 + 2, 1);
      D_T.finalize(index_offset + index);
    }
  }
}
}
