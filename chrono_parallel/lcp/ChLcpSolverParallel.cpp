#include "chrono_parallel/lcp/ChLcpSolverParallel.h"
#include "chrono_parallel/math/ChThrustLinearAlgebra.h"
#include "core/ChSpmatrix.h"
#include "physics/ChBody.h"
using namespace chrono;

ChLcpSolverParallel::ChLcpSolverParallel(ChParallelDataManager* dc) : data_manager(dc) {
  tolerance = 1e-7;
  record_violation_history = true;
  warm_start = false;
  residual = 0;
  solver = new ChSolverAPGD();
}

ChLcpSolverParallel::~ChLcpSolverParallel() {
  delete solver;
}

void ChLcpSolverParallel::ComputeInvMassMatrix() {
  LOG(INFO) << "ChLcpSolverParallel::ComputeInvMassMatrix()";
  data_manager->system_timer.start("ChLcpSolverParallel_M");
  uint num_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;
  uint num_fluid_bodies = data_manager->num_fluid_bodies;
  uint num_dof = data_manager->num_dof;
  bool use_full_inertia_tensor = data_manager->settings.solver.use_full_inertia_tensor;
  const custom_vector<real>& shaft_inr = data_manager->host_data.shaft_inr;

  const std::vector<ChBody*>* body_list = data_manager->body_list;
  const std::vector<ChLink*>* link_list = data_manager->link_list;
  const std::vector<ChPhysicsItem*>* other_physics_list = data_manager->other_physics_list;

  const DynamicVector<real>& hf = data_manager->host_data.hf;
  const DynamicVector<real>& v = data_manager->host_data.v;

  DynamicVector<real>& M_invk = data_manager->host_data.M_invk;
  CompressedMatrix<real>& M_inv = data_manager->host_data.M_inv;

  clear(M_inv);

  // Each rigid object has 3 mass entries and 9 inertia entries
  // Each shaft has one inertia entry
  M_inv.reserve(num_bodies * 12 + num_shafts * 1 + num_fluid_bodies * 3);
  // The mass matrix is square and each rigid body has 6 DOF
  // Shafts have one DOF
  M_inv.resize(num_dof, num_dof);

  for (int i = 0; i < num_bodies; i++) {
    if (data_manager->host_data.active_rigid[i]) {
      real inv_mass = 1.0 / body_list->at(i)->GetMass();
      ChMatrix33<>& body_inv_inr = body_list->at(i)->VariablesBody().GetBodyInvInertia();

      M_inv.append(i * 6 + 0, i * 6 + 0, inv_mass);
      M_inv.finalize(i * 6 + 0);
      M_inv.append(i * 6 + 1, i * 6 + 1, inv_mass);
      M_inv.finalize(i * 6 + 1);
      M_inv.append(i * 6 + 2, i * 6 + 2, inv_mass);
      M_inv.finalize(i * 6 + 2);

      M_inv.append(i * 6 + 3, i * 6 + 3, body_inv_inr.GetElement(0, 0));
      if (use_full_inertia_tensor) {
        M_inv.append(i * 6 + 3, i * 6 + 4, body_inv_inr.GetElement(0, 1));
        M_inv.append(i * 6 + 3, i * 6 + 5, body_inv_inr.GetElement(0, 2));
      }
      M_inv.finalize(i * 6 + 3);
      if (use_full_inertia_tensor) {
        M_inv.append(i * 6 + 4, i * 6 + 3, body_inv_inr.GetElement(1, 0));
      }
      M_inv.append(i * 6 + 4, i * 6 + 4, body_inv_inr.GetElement(1, 1));
      if (use_full_inertia_tensor) {
        M_inv.append(i * 6 + 4, i * 6 + 5, body_inv_inr.GetElement(1, 2));
      }
      M_inv.finalize(i * 6 + 4);
      if (use_full_inertia_tensor) {
        M_inv.append(i * 6 + 5, i * 6 + 3, body_inv_inr.GetElement(2, 0));
        M_inv.append(i * 6 + 5, i * 6 + 4, body_inv_inr.GetElement(2, 1));
      }
      M_inv.append(i * 6 + 5, i * 6 + 5, body_inv_inr.GetElement(2, 2));
      M_inv.finalize(i * 6 + 5);
    } else {
      M_inv.finalize(i * 6 + 0);
      M_inv.finalize(i * 6 + 1);
      M_inv.finalize(i * 6 + 2);
      M_inv.finalize(i * 6 + 3);
      M_inv.finalize(i * 6 + 4);
      M_inv.finalize(i * 6 + 5);
    }
  }
  int offset = num_bodies * 6;
  for (int i = 0; i < num_shafts; i++) {
    M_inv.append(offset + i, offset + i, shaft_inr[i]);
    M_inv.finalize(offset + i);
  }
  offset = num_bodies * 6 + num_shafts;
  real inv_fluid_mass = 1.0 / data_manager->settings.fluid.mass;
  for (int i = 0; i < num_fluid_bodies; i++) {
    M_inv.append(offset + i * 3 + 0, offset + i * 3 + 0, inv_fluid_mass);
    M_inv.finalize(offset + i * 3 + 0);
    M_inv.append(offset + i * 3 + 1, offset + i * 3 + 1, inv_fluid_mass);
    M_inv.finalize(offset + i * 3 + 1);
    M_inv.append(offset + i * 3 + 2, offset + i * 3 + 2, inv_fluid_mass);
    M_inv.finalize(offset + i * 3 + 2);
  }

  M_invk = v + M_inv * hf;
  data_manager->system_timer.stop("ChLcpSolverParallel_M");
}

void ChLcpSolverParallel::ComputeMassMatrix() {
  //LOG(INFO) << "ChLcpSolverParallel::ComputeMassMatrix()";
  uint num_bodies = data_manager->num_rigid_bodies;
  uint num_shafts = data_manager->num_shafts;
  uint num_fluid_bodies = data_manager->num_fluid_bodies;
  uint num_dof = data_manager->num_dof;
  bool use_full_inertia_tensor = data_manager->settings.solver.use_full_inertia_tensor;
  const custom_vector<real>& shaft_inr = data_manager->host_data.shaft_inr;

  const std::vector<ChBody*>* body_list = data_manager->body_list;
  const std::vector<ChLink*>* link_list = data_manager->link_list;
  const std::vector<ChPhysicsItem*>* other_physics_list = data_manager->other_physics_list;

  const DynamicVector<real>& hf = data_manager->host_data.hf;
  const DynamicVector<real>& v = data_manager->host_data.v;

  DynamicVector<real>& M_invk = data_manager->host_data.M_invk;
  CompressedMatrix<real>& M = data_manager->host_data.M;

  clear(M);

  // Each rigid object has 3 mass entries and 9 inertia entries
  // Each shaft has one inertia entry
  M.reserve(num_bodies * 12 + num_shafts * 1 + num_fluid_bodies * 3);
  // The mass matrix is square and each rigid body has 6 DOF
  // Shafts have one DOF
  M.resize(num_dof, num_dof);

  for (int i = 0; i < num_bodies; i++) {
    if (data_manager->host_data.active_rigid[i]) {
      real mass = body_list->at(i)->GetMass();
      ChMatrix33<>& body_inr = body_list->at(i)->VariablesBody().GetBodyInertia();

      M.append(i * 6 + 0, i * 6 + 0, mass);
      M.finalize(i * 6 + 0);
      M.append(i * 6 + 1, i * 6 + 1, mass);
      M.finalize(i * 6 + 1);
      M.append(i * 6 + 2, i * 6 + 2, mass);
      M.finalize(i * 6 + 2);

      M.append(i * 6 + 3, i * 6 + 3, body_inr.GetElement(0, 0));
      if (use_full_inertia_tensor) {
        M.append(i * 6 + 3, i * 6 + 4, body_inr.GetElement(0, 1));
        M.append(i * 6 + 3, i * 6 + 5, body_inr.GetElement(0, 2));
      }
      M.finalize(i * 6 + 3);
      if (use_full_inertia_tensor) {
        M.append(i * 6 + 4, i * 6 + 3, body_inr.GetElement(1, 0));
      }
      M.append(i * 6 + 4, i * 6 + 4, body_inr.GetElement(1, 1));
      if (use_full_inertia_tensor) {
        M.append(i * 6 + 4, i * 6 + 5, body_inr.GetElement(1, 2));
      }
      M.finalize(i * 6 + 4);
      if (use_full_inertia_tensor) {
        M.append(i * 6 + 5, i * 6 + 3, body_inr.GetElement(2, 0));
        M.append(i * 6 + 5, i * 6 + 4, body_inr.GetElement(2, 1));
      }
      M.append(i * 6 + 5, i * 6 + 5, body_inr.GetElement(2, 2));
      M.finalize(i * 6 + 5);
    } else {
      M.finalize(i * 6 + 0);
      M.finalize(i * 6 + 1);
      M.finalize(i * 6 + 2);
      M.finalize(i * 6 + 3);
      M.finalize(i * 6 + 4);
      M.finalize(i * 6 + 5);
    }
  }
  int offset = num_bodies * 6;
  for (int i = 0; i < num_shafts; i++) {
    M.append(offset + i, offset + i, 1.0 / shaft_inr[i]);
    M.finalize(offset + i);
  }
  offset = num_bodies * 6 + num_shafts;
  real fluid_mass = data_manager->settings.fluid.mass;
  for (int i = 0; i < num_fluid_bodies; i++) {
    M.append(offset + i * 3 + 0, offset + i * 3 + 0, fluid_mass);
    M.finalize(offset + i * 3 + 0);
    M.append(offset + i * 3 + 1, offset + i * 3 + 1, fluid_mass);
    M.finalize(offset + i * 3 + 1);
    M.append(offset + i * 3 + 2, offset + i * 3 + 2, fluid_mass);
    M.finalize(offset + i * 3 + 2);
  }
}

void ChLcpSolverParallel::PerformStabilization() {
  LOG(INFO) << "ChLcpSolverParallel::PerformStabilization";
  data_manager->system_timer.start("ChLcpSolverParallel_Stab");
  const DynamicVector<real>& R_full = data_manager->host_data.R_full;
  DynamicVector<real>& gamma = data_manager->host_data.gamma;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;

  if (data_manager->settings.solver.max_iteration_bilateral <= 0 || num_bilaterals <= 0) {
    return;
  }

  ConstSubVectorType R_b = blaze::subvector(R_full, num_unilaterals, num_bilaterals);
  SubVectorType gamma_b = blaze::subvector(gamma, num_unilaterals, num_bilaterals);

  solver->SolveStab(data_manager->settings.solver.max_iteration_bilateral, num_bilaterals, R_b, gamma_b);
  data_manager->system_timer.stop("ChLcpSolverParallel_Stab");
}
