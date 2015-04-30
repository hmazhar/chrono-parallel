#include "chrono_parallel/constraints/ChConstraintFluidFluid.h"
#include "chrono_parallel/constraints/ChConstraintFluidFluidUtils.h"
#include <thrust/iterator/constant_iterator.h>

namespace chrono {

void ChConstraintFluidFluid::Project(real* gamma) {
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
  } else {
#pragma omp parallel for
    for (int index = 0; index < num_fluid_contacts; index++) {
      real cohesion = data_manager->settings.fluid.cohesion;
      real gam = gamma[index_offset + index];
      gam = gam + cohesion;
      gam = gam < 0 ? 0 : gam - cohesion;
      gamma[index_offset + index] = gam;
    }
  }
}

void ChConstraintFluidFluid::Build_D_Rigid() {
  LOG(INFO) << "ChConstraintFluidFluid::Build_D_Rigid";

  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
  host_vector<real3>& pos_fluid = data_manager->host_data.pos_fluid;
  real step_size = data_manager->settings.step_size;
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

void ChConstraintFluidFluid::Density_Fluid() {
  LOG(INFO) << "ChConstraintFluidFluid::Density_Fluid";
  real mass_fluid = data_manager->settings.fluid.mass;
  real h = data_manager->settings.fluid.kernel_radius;
  host_vector<real3>& vel = data_manager->host_data.vel_fluid;
  host_vector<real3>& pos = data_manager->host_data.pos_fluid;
  host_vector<real>& density = data_manager->host_data.den_fluid;
// density of the particle itself

#pragma omp parallel for
  for (int i = 0; i < last_body; i++) {
    uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
    int2 bid;
    bid.x = fluid_contact_idA_start[i];
    real dens = 0;
    real corr = 0;
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      real3 xij = (pos[bid.x] - pos[bid.y]);
      real dist = length(xij);
      dens += mass_fluid * KERNEL(dist, h);
      // corr += k * pow(KERNEL(dist, 2 * h) / KERNEL(dq, 2 * h), n);
    }
    density[bid.x] = dens;
    // std::cout << dens << " " << corr << std::endl;
  }
}

void ChConstraintFluidFluid::Normalize_Density_Fluid() {
  real h = data_manager->settings.fluid.kernel_radius;
  host_vector<real3>& vel = data_manager->host_data.vel_fluid;
  host_vector<real3>& pos = data_manager->host_data.pos_fluid;
  real mass_fluid = data_manager->settings.fluid.mass;
  host_vector<real>& density = data_manager->host_data.den_fluid;
  real single_density = mass_fluid * KERNEL(0, h);
// custom_vector<real> density_denom = density;
#pragma omp parallel for
  for (int i = 0; i < last_body; i++) {
    uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
    int2 bid;
    bid.x = fluid_contact_idA_start[i];
    real dens = 0;
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      if (bid.x == bid.y) {
        continue;
      }
      real dist = length(pos[bid.x] - pos[bid.y]);
      dens += (mass_fluid / density[bid.y]) * KERNEL(dist, h);
    }
    density[bid.x] = density[bid.x] / (dens);
  }
}
void ChConstraintFluidFluid::Build_D_Fluid() {
  LOG(INFO) << "ChConstraintFluidFluid::Build_D_Fluid";
  //  if (num_fluid_contacts <= 0) {
  //    return;
  //  }
  real viscosity_fluid = data_manager->settings.fluid.viscosity;
  real mass_fluid = data_manager->settings.fluid.mass;
  real h = data_manager->settings.fluid.kernel_radius;
  real envel = data_manager->settings.collision.collision_envelope;
  real density_fluid = data_manager->settings.fluid.density;
  real inv_density = 1.0 / density_fluid;
  real mass_over_density = mass_fluid * inv_density;
  real eta = .01;

  host_vector<real>& density = data_manager->host_data.den_fluid;
  host_vector<real3>& vel = data_manager->host_data.vel_fluid;
  host_vector<real3>& pos = data_manager->host_data.pos_fluid;

  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;

  // shear_tensor.clear();
  // shear_tensor.resize(num_fluid_bodies);

  //=======COMPUTE DENSITY OF FLUID
  density.resize(num_fluid_bodies);
  // Density_Fluid();
  // Normalize_Density_Fluid();
  den_con.resize(num_fluid_contacts * 2 + num_fluid_bodies);
  //  den_vec.resize(num_fluid_contacts * 2 + num_fluid_bodies);
  //#pragma omp parallel for
  //  for (int i = 0; i < fluid_contact_idB.size(); i++) {
  //    int2 bid;
  //    bid.x = fluid_contact_idA[i];
  //    bid.y = fluid_contact_idB[i];
  //    real3 xij = (pos[bid.x] - pos[bid.y]);
  //    real dist = length(xij);
  //    den_vec[i] = R4(dist, xij.x, xij.y, xij.z);
  //  }
  //
  //  real spiky(const real& dist, const real& h) {
  //    return (dist <= h) * 15.0 / (F_PI * pow(h, 6)) * pow(h - dist, 3);
  //  }
  //  real3 grad_spiky(const real3& dist, const real d, const real& h) {
  //    return (d <= h) * -45.0 / (F_PI * pow(h, 6)) * pow(h - d, 2) * dist;
  //  }
  const real inv_f_pi_h_6 = 1.0 / (F_PI * pow(h, 6));
  const real h_3 = h * h * h;
#pragma omp parallel for
  for (int i = 0; i < last_body; i++) {
    int2 bid;
    uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
    bid.x = fluid_contact_idA_start[i];
    real3 posa = pos[bid.x];
    real3 diag = 0;
    int diag_index = 0;
    real dens = 0;
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      if (bid.x == bid.y) {
        diag_index = index;
        dens += mass_fluid * KERNEL(0, h);
        continue;
      }
      real3 xij = (posa - pos[bid.y]);
      real dist = length(xij);
      real3 off_diag = mass_over_density * -45.0 * inv_f_pi_h_6 * pow(h - dist, 2) * xij;
      den_con[index] = off_diag;
      diag += off_diag;
      dens += mass_fluid * KERNEL(dist, h);
    }
    den_con[diag_index] = -diag;
    density[bid.x] = dens;
    // std::cout<<diag.x<<" "<<diag.y<<" "<<diag.z<<"\n";
  }

  LOG(INFO) << "ChConstraintFluidFluid::JACOBIAN OF FLUID";
  for (int i = 0; i < last_body; i++) {
    uint start = fluid_start_index[i];
    uint end = fluid_start_index[i + 1];
    int2 bid;
    bid.x = fluid_contact_idA_start[i];
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      D_T.append(index_offset + bid.x, body_offset + bid.y * 3 + 0, den_con[index].x);
      D_T.append(index_offset + bid.x, body_offset + bid.y * 3 + 1, den_con[index].y);
      D_T.append(index_offset + bid.x, body_offset + bid.y * 3 + 2, den_con[index].z);
    }
    D_T.finalize(index_offset + bid.x);
  }

  //=======COMPUTE JACOBIAN OF FLUID
  //#pragma omp parallel for
  //  for (int i = 0; i < last_body; i++) {
  //    uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
  //    int2 bid;
  //    bid.x = fluid_contact_idA[i];
  //    real3 diagonal = 0, off_diag;
  //    for (int index = start; index < end; index++) {
  //      bid.y = fluid_contact_idB[index];
  //      if (bid.x != bid.y) {
  //        // xij is also the normal vector between the points
  //        real3 xij = (pos[bid.x] - pos[bid.y]);
  //        real dist = length(xij);
  //        real3 grad_kernel = GRAD_KERNEL(xij, dist, h);
  //        off_diag = (mass_over_density)*grad_kernel;
  //        D_T.set(index_offset + bid.x, body_offset + bid.y * 3 + 0, off_diag.x);
  //        D_T.set(index_offset + bid.x, body_offset + bid.y * 3 + 1, off_diag.y);
  //        D_T.set(index_offset + bid.x, body_offset + bid.y * 3 + 2, off_diag.z);
  //        diagonal += off_diag;
  //      }
  //    }
  //    D_T.set(index_offset + bid.x, body_offset + bid.x * 3 + 0, -diagonal.x);
  //    D_T.set(index_offset + bid.x, body_offset + bid.x * 3 + 1, -diagonal.y);
  //    D_T.set(index_offset + bid.x, body_offset + bid.x * 3 + 2, -diagonal.z);
  //  }
}
void ChConstraintFluidFluid::Build_D() {
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
    Build_D_Fluid();
  } else {
    Build_D_Rigid();
  }
}

void ChConstraintFluidFluid::Build_b() {
  real step_size = data_manager->settings.step_size;
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
    SubVectorType b_sub = blaze::subvector(data_manager->host_data.b, index_offset, num_fluid_bodies);
    SubVectorType v_sub = blaze::subvector(data_manager->host_data.v, body_offset, num_fluid_bodies * 3);
    CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
    SubMatrixType D_T_sub = submatrix(D_T, index_offset, body_offset, num_fluid_bodies, num_fluid_bodies * 3);

    real density_fluid = data_manager->settings.fluid.density;
    host_vector<real>& density = data_manager->host_data.den_fluid;
    host_vector<real3>& pos = data_manager->host_data.pos_fluid;
    DynamicVector<real> g(num_fluid_bodies);
    real epsilon = data_manager->settings.fluid.epsilon;
    real tau = data_manager->settings.fluid.tau;
    real h = data_manager->settings.fluid.kernel_radius;
    real zeta = 1.0 / (1.0 + 4.0 * tau / h);

#pragma omp parallel for
    for (int index = 0; index < num_fluid_bodies; index++) {
      // if not normalized
      // g[index] = (density[index] - density_fluid) / density_fluid;
      g[index] = density[index] / density_fluid - 1.0;
      // if normalized:
      // g[index] = (density[index] - 1) / 1;

      // std::cout << g[index] << " " << density[index] << std::endl;
    }
    //b_sub = -4.0 / step_size * zeta * g + zeta * D_T_sub * v_sub;
    // b_sub = -g + zeta * D_T_sub * v_sub;
     b_sub = -g;  // + D_T_sub * v_sub;
  } else {
    SubVectorType b_sub = blaze::subvector(data_manager->host_data.b, index_offset, num_fluid_contacts);
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
      real bi = 0;
      if (data_manager->settings.solver.alpha > 0) {
        bi = inv_hpa * depth;
      } else {
        if (data_manager->settings.solver.contact_recovery_speed < 0) {
          bi = real(1.0) / step_size * depth;
        } else {
          bi = std::max(real(1.0) / step_size * depth, -data_manager->settings.fluid.contact_recovery_speed);
        }
      }
      b_sub[index] = bi;
    }
  }
}
void ChConstraintFluidFluid::Build_E() {
  DynamicVector<real>& E = data_manager->host_data.E;
  real step_size = data_manager->settings.step_size;
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
    if (num_fluid_bodies <= 0) {
      return;
    }

    real epsilon = data_manager->settings.fluid.epsilon;
    real tau = data_manager->settings.fluid.tau;
    real h = data_manager->settings.fluid.kernel_radius;

    real zeta = 1.0 / (1.0 + 4.0 * tau / step_size);
    real compliance = 4.0 / (step_size * step_size) * (epsilon * zeta);
#pragma omp parallel for
    for (int index = 0; index < num_fluid_bodies; index++) {
      E[index_offset + index] = compliance;
    }

    //#pragma omp parallel for
    //    for (int index = 0; index < num_fluid_bodies; index++) {
    //      E[index_offset + num_fluid_bodies + index * 3 + 0] = 0;  // 1.0/step_size * 1;    // compliance;
    //      E[index_offset + num_fluid_bodies + index * 3 + 1] = 0;  // 1.0/step_size * 1;    // compliance;
    //      E[index_offset + num_fluid_bodies + index * 3 + 2] = 0;  // 1.0/step_size * 1;    // compliance;
    //    }

  } else {
    if (num_fluid_contacts <= 0) {
      return;
    }
    real compliance = data_manager->settings.fluid.compliance;
    real inv_hhpa = 1.0 / (step_size * (step_size + data_manager->settings.solver.alpha));
#pragma omp parallel for
    for (int index = 0; index < num_fluid_contacts; index++) {
      E[index_offset + index] = inv_hhpa * compliance;
    }
  }
}
void ChConstraintFluidFluid::GenerateSparsityRigid() {
  LOG(INFO) << "ChConstraintFluidFluid::GenerateSparsityRigid";

  custom_vector<int2>& bids = data_manager->host_data.bids_fluid_fluid;

  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
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
void ChConstraintFluidFluid::GenerateSparsityFluid() {
  LOG(INFO) << "ChConstraintFluidFluid::GenerateSparsityFluid";
  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;

  // density constraints
  for (int i = 0; i < last_body; i++) {
    uint start = fluid_start_index[i];
    uint end = fluid_start_index[i + 1];
    int2 bid;
    bid.x = fluid_contact_idA_start[i];
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      D_T.append(index_offset + bid.x, body_offset + bid.y * 3 + 0, 1);
      D_T.append(index_offset + bid.x, body_offset + bid.y * 3 + 1, 1);
      D_T.append(index_offset + bid.x, body_offset + bid.y * 3 + 2, 1);
    }
    D_T.finalize(index_offset + bid.x);
  }
  // Add more entries for viscosity
  // Code is repeated because there are three rows per viscosity constraint

  //  for (int i = 0; i < last_body; i++) {
  //    uint start = (i == 0) ? 0 : fluid_start_index[i - 1];
  //    uint end = fluid_start_index[i];
  //    int2 bid;
  //    bid.x = fluid_contact_idA[i];
  //    for (int index = start; index < end; index++) {
  //      bid.y = fluid_contact_idB[index];
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.y * 3 + 0, 1);
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.y * 3 + 1, 1);
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.y * 3 + 2, 1);
  //    }
  //    D_T.finalize(index_offset + num_fluid_bodies + bid.x * 3 + 0);
  //
  //    for (int index = start; index < end; index++) {
  //      bid.y = fluid_contact_idB[index];
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.y * 3 + 0, 1);
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.y * 3 + 1, 1);
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.y * 3 + 2, 1);
  //    }
  //    D_T.finalize(index_offset + num_fluid_bodies + bid.x * 3 + 1);
  //
  //    for (int index = start; index < end; index++) {
  //      bid.y = fluid_contact_idB[index];
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.y * 3 + 0, 1);
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.y * 3 + 1, 1);
  //      D_T.append(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.y * 3 + 2, 1);
  //    }
  //    D_T.finalize(index_offset + num_fluid_bodies + bid.x * 3 + 2);
  //  }
}
void ChConstraintFluidFluid::GenerateSparsity() {
  if (data_manager->num_fluid_contacts <= 0) {
    return;
  }

  if (data_manager->settings.fluid.fluid_is_rigid == false) {
    DetermineNeighbors();
    // GenerateSparsityFluid();
  } else {
    GenerateSparsityRigid();
  }
}

void ChConstraintFluidFluid::DetermineNeighbors() {
  LOG(INFO) << "ChConstraintFluidFluid::DetermineNeighbors";
  // get a reference to the contact body ID data
  host_vector<int2>& bids = data_manager->host_data.bids_fluid_fluid;
  fluid_contact_idA.resize(num_fluid_contacts * 2 + num_fluid_bodies);
  fluid_contact_idB.resize(num_fluid_contacts * 2 + num_fluid_bodies);
  fluid_contact_idA_start.resize(num_fluid_contacts * 2 + num_fluid_bodies);

// For each contact in the list that is a fluid contact
#pragma omp parallel for
  for (int index = 0; index < num_fluid_bodies; index++) {
    fluid_contact_idA[index] = index;
    fluid_contact_idB[index] = index;
  }
#pragma omp parallel for
  for (int index = 0; index < num_fluid_contacts; index++) {
    int2 body_id = bids[index];
    fluid_contact_idA[index + num_fluid_bodies] = body_id.x;
    fluid_contact_idA[index + num_fluid_bodies + num_fluid_contacts] = body_id.y;

    fluid_contact_idB[index + num_fluid_bodies] = body_id.y;
    fluid_contact_idB[index + num_fluid_bodies + num_fluid_contacts] = body_id.x;
  }

  Thrust_Sort_By_Key(fluid_contact_idB, fluid_contact_idA);
  Thrust_Sort_By_Key(fluid_contact_idA, fluid_contact_idB);

  fluid_start_index.resize(num_fluid_bodies);

  last_body = Thrust_Reduce_By_Key(fluid_contact_idA, fluid_contact_idA_start, fluid_start_index);
  fluid_start_index.resize(last_body + 1);
  fluid_start_index[last_body] = 0;
  Thrust_Exclusive_Scan(fluid_start_index);
}
void ChConstraintFluidFluid::ArtificialPressure() {
  if (data_manager->settings.fluid.artificial_pressure == false) {
    return;
  }
  if (data_manager->settings.fluid.fluid_is_rigid == false) {
    host_vector<real3>& pos = data_manager->host_data.pos_fluid;
    real mass_fluid = data_manager->settings.fluid.mass;
    real h = data_manager->settings.fluid.kernel_radius;
    real k = data_manager->settings.fluid.artificial_pressure_k;
    real dq = data_manager->settings.fluid.artificial_pressure_dq;
    real n = data_manager->settings.fluid.artificial_pressure_n;
#pragma omp parallel for
    for (int i = 0; i < last_body; i++) {
      uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
      int2 bid;
      bid.x = fluid_contact_idA_start[i];
      real corr = 0;
      for (int index = start; index < end; index++) {
        bid.y = fluid_contact_idB[index];
        real3 xij = (pos[bid.x] - pos[bid.y]);
        real dist = length(xij);
        corr += k * pow(KERNEL(dist, 2 * h) / KERNEL(dq, h), n);
      }
      // std::cout << gamma[index_offset + bid.x] << " " << corr << std::endl;
      data_manager->host_data.gamma[index_offset + bid.x] += corr;
    }
  }
}
}
