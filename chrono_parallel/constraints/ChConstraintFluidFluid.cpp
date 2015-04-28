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

void ChConstraintFluidFluid::Build_D_Fluid() {
  LOG(INFO) << "ChConstraintFluidFluid::Build_D_Fluid";
  if (num_fluid_contacts <= 0) {
    return;
  }
  real viscosity_fluid = data_manager->settings.fluid.viscosity;
  real mass_fluid = data_manager->settings.fluid.mass;
  real h = data_manager->settings.fluid.kernel_radius;
  real density_fluid = data_manager->settings.fluid.density;
  real inv_density = 1.0 / density_fluid;
  real mass_over_density = mass_fluid * inv_density;
  real eta = .01;

  // density of the particle itself
  real single_density = mass_fluid * KERNEL(0, h);

  host_vector<real>& density = data_manager->host_data.den_fluid;
  host_vector<real3>& vel = data_manager->host_data.vel_fluid;
  host_vector<real3>& pos = data_manager->host_data.pos_fluid;

  CompressedMatrix<real>& D_T = data_manager->host_data.D_T;

  shear_tensor.clear();
  shear_tensor.resize(num_fluid_bodies);
  density.resize(num_fluid_bodies);
  LOG(INFO) << "ChConstraintFluidFluid::DENSITY OF FLUID";
//=======COMPUTE DENSITY OF FLUID
#pragma omp parallel for
  for (int i = 0; i < last_body; i++) {
    uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
    int2 bid;
    bid.x = fluid_contact_idA[i];
    real dens = 0;
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      if (bid.x != bid.y) {
        real3 vij = (vel[bid.x] - vel[bid.y]);
        real3 xij = (pos[bid.x] - pos[bid.y]);
        real dist = length(xij);
        real spline_kernel = KERNEL(dist, h);
        dens += (mass_fluid * spline_kernel);
      }
    }
    density[bid.x] = dens + single_density;
  }

  LOG(INFO) << "ChConstraintFluidFluid::JACOBIAN OF FLUID";
//=======COMPUTE JACOBIAN OF FLUID
#pragma omp parallel for
  for (int i = 0; i < last_body; i++) {
    uint start = fluid_start_index[i], end = fluid_start_index[i + 1];
    int2 bid;
    bid.x = fluid_contact_idA[i];
    real3 diagonal = 0, off_diag;
    //M33 diagonal_matrix;
    for (int index = start; index < end; index++) {
      bid.y = fluid_contact_idB[index];
      if (bid.x != bid.y) {
        // xij is also the normal vector between the points
        real3 xij = (pos[bid.x] - pos[bid.y]);
        real dist = length(xij);
        // real3 xij_hat = xij / dist;
        // real spline_kernel = KERNEL(dist, h);
        real3 grad_kernel = GRAD_KERNEL(xij, dist, h);
        // real3 grad2_kernel = GRAD2_KERNEL(xij, dist, h);
        // n = n / dist;
        // real correction = -k * pow(spline_kernel / spline_dq, N);
        off_diag = (mass_over_density)*grad_kernel;
        D_T.set(index_offset + bid.x, body_offset + bid.y * 3 + 0, off_diag.x);
        D_T.set(index_offset + bid.x, body_offset + bid.y * 3 + 1, off_diag.y);
        D_T.set(index_offset + bid.x, body_offset + bid.y * 3 + 2, off_diag.z);
        diagonal += off_diag;
        //
        //        // The scalar term in the viscosity term
        //        real norm_shear_a, norm_shear_b;
        //        real visca, viscb;
        //        real corr = 1e-5;
        //
        //        real ta = Trace(shear_tensor[bid.x]);
        //        real tb = Trace(shear_tensor[bid.y]);
        //        // norm_shear_a = sqrt(Trace(shear_tensor[bid.x] * Transpose(shear_tensor[bid.x])));    //
        //        // TensorNorm(shear_tensor[bid.x]);
        //        // norm_shear_b = sqrt(Trace(shear_tensor[bid.y] * Transpose(shear_tensor[bid.y])));    //
        //        // TensorNorm(shear_tensor[bid.y]);
        //        norm_shear_a = sqrt(ta * ta);
        //        norm_shear_b = sqrt(tb * tb);
        //
        //        // visca = (mu_b + tau_b / (corr + norm_shear_a));
        //        // viscb = (mu_b + tau_b / (corr + norm_shear_b));
        //
        //        // newtonian
        //        visca = viscosity_fluid;
        //        viscb = viscosity_fluid;
        //
        //        // Power law fluid
        //        real n = 0.5;
        //        real J = 15000;
        //
        //        // visca = (1-exp(-(J+1)*(corr+norm_shear_a)))*(pow(corr+norm_shear_a, n-1)+(1.0/(corr
        //        +norm_shear_a)));
        //        // viscb = (1-exp(-(J+1)*(corr+norm_shear_b)))*(pow(corr+norm_shear_b, n-1)+(1.0/(corr
        //        +norm_shear_b)));
        //        // visca = K * pow(norm_shear_a, n - 1.0);
        //        // viscb = K * pow(norm_shear_b, n - 1.0);
        //
        //        // Bingham FLuid
        //        real mu_b = 29.8;
        //        real tau_b = 34.0;
        //        // visca = (mu_b + tau_b / (corr + norm_shear_a));
        //        // viscb = (mu_b + tau_b / (corr + norm_shear_b));
        //
        //        // Clamp to prevent numerical issues
        //        // visca = clamp(visca, 0.0f, 30000.0f);
        //        // viscb = clamp(viscb, 0.0f, 30000.0f);
        //
        //        //
        //        https://github.com/oysteinkrog/gpusphsim/blob/5ec57973e387bfa6ace5ad7ac55ebcba508587fe/SPHSimLib/K_SnowSPH_Step2.inl
        //        //
        //        https://github.com/JimBrouzoulis/OOFEM_LargeDef/blob/9f00fc26814e9fbd00a32f9ca18be8b4e3c15912/src/fm/binghamfluid2.C
        //        // https://github.com/arousta/RheolefCode/tree/d5e1cde732ba26a6ead9d7762b639e44154761e2/rheolef
        //        // https://github.com/anassalamah/nlp-project
        //        real scalar = -mass_fluid * mass_fluid * (8.0 / (density[bid.x] + density[bid.y])) * (visca + viscb) *
        //        1.0 /
        //                      (h * ((dist * dist) / (h * h) + eta * eta));
        //
        //        // scalar = -(1000000) * mass_fluid*mass_fluid/(density[bid.x]+density[bid.y]);
        //
        //        // printf("%f %f %f\n", visca, viscb, scalar);
        //
        //        M33 matrix = VectorxVector(xij, grad_kernel) * scalar;
        //
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.y * 3 + 0) =
        //        matrix.U.x;
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.y * 3 + 1) =
        //        matrix.V.x;
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.y * 3 + 2) =
        //        matrix.W.x;
        //        //
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.y * 3 + 0) =
        //        matrix.U.y;
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.y * 3 + 1) =
        //        matrix.V.y;
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.y * 3 + 2) =
        //        matrix.W.y;
        //        //
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.y * 3 + 0) =
        //        matrix.U.z;
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.y * 3 + 1) =
        //        matrix.V.z;
        //        //        D_T(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.y * 3 + 2) =
        //        matrix.W.z;
        //        //
        //        diagonal_matrix = diagonal_matrix + matrix;
      }
      // else {
      // dens += mass_fluid * KERNEL(0, h);
      //}
    }
    D_T.set(index_offset + bid.x, body_offset + bid.x * 3 + 0, -diagonal.x);
    D_T.set(index_offset + bid.x, body_offset + bid.x * 3 + 1, -diagonal.y);
    D_T.set(index_offset + bid.x, body_offset + bid.x * 3 + 2, -diagonal.z);

    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.x * 3 + 0) = -diagonal_matrix.U.x;
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.x * 3 + 1) = -diagonal_matrix.V.x;
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 0, body_offset + bid.x * 3 + 2) = -diagonal_matrix.W.x;
    //
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.x * 3 + 0) = -diagonal_matrix.U.y;
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.x * 3 + 1) = -diagonal_matrix.V.y;
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 1, body_offset + bid.x * 3 + 2) = -diagonal_matrix.W.y;
    //
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.x * 3 + 0) = -diagonal_matrix.U.z;
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.x * 3 + 1) = -diagonal_matrix.V.z;
    //    D_T(index_offset + num_fluid_bodies + bid.x * 3 + 2, body_offset + bid.x * 3 + 2) = -diagonal_matrix.W.z;
  }
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
    real zeta = 1.0 / (1.0 + 4.0 * tau / step_size);

#pragma omp parallel for
    for (int index = 0; index < num_fluid_bodies; index++) {
      g[index] = -(density[index] - density_fluid) / density_fluid;
    }
    b_sub = g + zeta * D_T_sub * v_sub;

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
      E[index_offset + index] = 0;  // compliance;
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
    bid.x = fluid_contact_idA[i];
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
    GenerateSparsityFluid();
  } else {
    GenerateSparsityRigid();
  }
}

void ChConstraintFluidFluid::DetermineNeighbors() {
  LOG(INFO) << "ChConstraintFluidFluid::DetermineNeighbors";

  // get a reference to the contact body ID data
  host_vector<int2>& bids = data_manager->host_data.bids_fluid_fluid;

  // For each fluid particle determine which ones it is in contact with
  // To do this, take the contact pair list, make a copy of it so that the body
  // indices are flipped
  // Then add this to the first list
  // Sort the list
  // Expand the list back out

  fluid_contact_idA.resize(num_fluid_contacts * 2 + num_fluid_bodies);
  fluid_contact_idB.resize(num_fluid_contacts * 2 + num_fluid_bodies);

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
    //
    fluid_contact_idB[index + num_fluid_bodies] = body_id.y;
    fluid_contact_idB[index + num_fluid_bodies + num_fluid_contacts] = body_id.x;
  }

  thrust::sort_by_key(fluid_contact_idB.begin(), fluid_contact_idB.end(), fluid_contact_idA.begin());
  thrust::sort_by_key(fluid_contact_idA.begin(), fluid_contact_idA.end(), fluid_contact_idB.begin());

  fluid_start_index.resize(num_fluid_bodies + 1);
  fluid_start_index[num_fluid_bodies] = 0;
  last_body = Thrust_Reduce_By_Key(fluid_contact_idA, fluid_contact_idA, fluid_start_index);
  fluid_start_index.resize(last_body);
  Thrust_Exclusive_Scan(fluid_start_index);
}
}
