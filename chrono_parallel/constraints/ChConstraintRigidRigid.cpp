#include <algorithm>
#include <limits>

#include "chrono_parallel/ChConfigParallel.h"
#include "chrono_parallel/constraints/ChConstraintRigidRigid.h"
#include "chrono_parallel/constraints/ChConstraintUtils.h"
#include "chrono_parallel/math/quartic.h"
#include <thrust/iterator/constant_iterator.h>

using namespace chrono;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bool Cone_generalized(real& gamma_n, real& gamma_u, real& gamma_v, const real& mu) {
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

void Cone_single(real& gamma_n, real& gamma_s, const real& mu) {
  real f_tang = fabs(gamma_s);

  // inside upper cone? keep untouched!
  if (f_tang < (mu * gamma_n)) {
    return;
  }

  // inside lower cone? reset  normal,u,v to zero!
  if ((f_tang) < -(1.0 / mu) * gamma_n || (fabs(gamma_n) < 10e-15)) {
    gamma_n = 0;
    gamma_s = 0;
    return;
  }

  // remaining case: project orthogonally to generator segment of upper cone
  gamma_n = (f_tang * mu + gamma_n) / (mu * mu + 1);
  real tproj_div_t = (gamma_n * mu) / f_tang;
  gamma_s *= tproj_div_t;
}

void ChConstraintRigidRigid::func_Project_normal(int index, const int2* ids, const real* cohesion, real* gamma) {
  real gamma_x = gamma[index * 1 + 0];
  int2 body_id = ids[index];
  real coh = cohesion[index];

  gamma_x += coh;
  gamma_x = gamma_x < 0 ? 0 : gamma_x - coh;
  gamma[index * 1 + 0] = gamma_x;
  if (data_manager->settings.solver.solver_mode == SLIDING) {
    gamma[data_manager->num_rigid_contacts + index * 2 + 0] = 0;
    gamma[data_manager->num_rigid_contacts + index * 2 + 1] = 0;
  }
  if (data_manager->settings.solver.solver_mode == SPINNING) {
    gamma[3 * data_manager->num_rigid_contacts + index * 3 + 0] = 0;
    gamma[3 * data_manager->num_rigid_contacts + index * 3 + 1] = 0;
    gamma[3 * data_manager->num_rigid_contacts + index * 3 + 2] = 0;
  }
}

void ChConstraintRigidRigid::func_Project_sliding(int index,
                                                  const int2* ids,
                                                  const real3* fric,
                                                  const real* cohesion,
                                                  real* gam) {
  real3 gamma;
  gamma.x = gam[index * 1 + 0];
  gamma.y = gam[data_manager->num_rigid_contacts + index * 2 + 0];
  gamma.z = gam[data_manager->num_rigid_contacts + index * 2 + 1];

  //  if (data_manager->settings.solver.solver_mode == SPINNING) {
  //    gam[3 * data_manager->num_contacts + index * 3 + 0] = 0;
  //    gam[3 * data_manager->num_contacts + index * 3 + 1] = 0;
  //    gam[3 * data_manager->num_contacts + index * 3 + 2] = 0;
  //  }

  real coh = cohesion[index];
  gamma.x += coh;

  real mu = fric[index].x;
  if (mu == 0) {
    gamma.x = gamma.x < 0 ? 0 : gamma.x - coh;
    gamma.y = gamma.z = 0;

    gam[index * 1 + 0] = gamma.x;
    gam[data_manager->num_rigid_contacts + index * 2 + 0] = gamma.y;
    gam[data_manager->num_rigid_contacts + index * 2 + 1] = gamma.z;

    return;
  }

  if (Cone_generalized(gamma.x, gamma.y, gamma.z, mu)) {
  }

  gam[index * 1 + 0] = gamma.x - coh;
  gam[data_manager->num_rigid_contacts + index * 2 + 0] = gamma.y;
  gam[data_manager->num_rigid_contacts + index * 2 + 1] = gamma.z;
}
void ChConstraintRigidRigid::func_Project_spinning(int index, const int2* ids, const real3* fric, real* gam) {
  // real3 gamma_roll = R3(0);
  real rollingfriction = fric[index].y;
  real spinningfriction = fric[index].z;

  //	if(rollingfriction||spinningfriction){
  //		gam[index + number_of_contacts * 1] = 0;
  //		gam[index + number_of_contacts * 2] = 0;
  //	}

  real gamma_n = fabs(gam[index * 1 + 0]);
  real gamma_s = gam[3 * data_manager->num_rigid_contacts + index * 3 + 0];
  real gamma_tu = gam[3 * data_manager->num_rigid_contacts + index * 3 + 1];
  real gamma_tv = gam[3 * data_manager->num_rigid_contacts + index * 3 + 2];

  if (spinningfriction == 0) {
    gamma_s = 0;

  } else {
    Cone_single(gamma_n, gamma_s, spinningfriction);
  }

  if (rollingfriction == 0) {
    gamma_tu = 0;
    gamma_tv = 0;
    //		if (gamma_n < 0) {
    //			gamma_n = 0;
    //		}
  } else {
    Cone_generalized(gamma_n, gamma_tu, gamma_tv, rollingfriction);
  }
  // gam[index + number_of_contacts * 0] = gamma_n;
  gam[3 * data_manager->num_rigid_contacts + index * 3 + 0] = gamma_s;
  gam[3 * data_manager->num_rigid_contacts + index * 3 + 1] = gamma_tu;
  gam[3 * data_manager->num_rigid_contacts + index * 3 + 2] = gamma_tv;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void ChConstraintRigidRigid::host_Project_single(int index, int2* ids, real3* friction, real* cohesion, real* gamma) {
  // always project normal
  switch (data_manager->settings.solver.local_solver_mode) {
    case NORMAL: {
      func_Project_normal(index, ids, cohesion, gamma);
    } break;

    case SLIDING: {
      func_Project_sliding(index, ids, friction, cohesion, gamma);
    } break;

    case SPINNING: {
      func_Project_sliding(index, ids, friction, cohesion, gamma);
      func_Project_spinning(index, ids, friction, gamma);
    } break;
  }
}

void ChConstraintRigidRigid::Project(real* gamma) {
  const thrust::host_vector<int2>& bids = data_manager->host_data.bids_rigid_rigid;
  const thrust::host_vector<real3>& friction = data_manager->host_data.fric_rigid_rigid;
  const thrust::host_vector<real>& cohesion = data_manager->host_data.coh_rigid_rigid;

  switch (data_manager->settings.solver.local_solver_mode) {
    case NORMAL: {
#pragma omp parallel for
      for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
        func_Project_normal(index, bids.data(), cohesion.data(), gamma);
      }
    } break;

    case SLIDING: {
#pragma omp parallel for
      for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
        func_Project_sliding(index, bids.data(), friction.data(), cohesion.data(), gamma);
      }
    } break;

    case SPINNING: {
#pragma omp parallel for
      for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
        func_Project_sliding(index, bids.data(), friction.data(), cohesion.data(), gamma);
        func_Project_spinning(index, bids.data(), friction.data(), gamma);
      }
    } break;
  }
}
void ChConstraintRigidRigid::Project_Single(int index, real* gamma) {
  thrust::host_vector<int2>& bids = data_manager->host_data.bids_rigid_rigid;
  thrust::host_vector<real3>& friction = data_manager->host_data.fric_rigid_rigid;
  thrust::host_vector<real>& cohesion = data_manager->host_data.coh_rigid_rigid;

  host_Project_single(index, bids.data(), friction.data(), cohesion.data(), gamma);
}

void ChConstraintRigidRigid::Build_b() {
  if (data_manager->num_rigid_contacts <= 0) {
    return;
  }

#pragma omp parallel for
  for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
    real bi = 0;
    real depth = data_manager->host_data.dpth_rigid_rigid[index];

    if (data_manager->settings.solver.alpha > 0) {
      bi = inv_hpa * depth;
    } else if (data_manager->settings.solver.contact_recovery_speed < 0) {
      bi = inv_h * depth;
    } else {
      bi = std::max(inv_h * depth, -data_manager->settings.solver.contact_recovery_speed);
      // bi = std::min(bi, inv_h * data_manager->settings.solver.cohesion_epsilon);
    }

    data_manager->host_data.b[index * 1 + 0] = bi;
  }
}

void ChConstraintRigidRigid::Build_s() {
  if (data_manager->num_rigid_contacts <= 0) {
    return;
  }

  if (data_manager->settings.solver.solver_mode == NORMAL) {
    return;
  }

  int2* ids = data_manager->host_data.bids_rigid_rigid.data();
  const SubMatrixType& D_t_T = _DTT_;
  DynamicVector<real> v_new;

  const DynamicVector<real>& M_invk = data_manager->host_data.M_invk;
  const DynamicVector<real>& gamma = data_manager->host_data.gamma;

  const SubMatrixType& M_invD_n = _MINVDN_;
  const SubMatrixType& M_invD_t = _MINVDT_;
  const SubMatrixType& M_invD_s = _MINVDS_;
  const SubMatrixType& M_invD_b = _MINVDB_;
  const CompressedMatrix<real>& M_invD = data_manager->host_data.M_invD;

  uint num_contacts = data_manager->num_rigid_contacts;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint num_bilaterals = data_manager->num_bilaterals;

  blaze::DenseSubvector<const DynamicVector<real> > gamma_b = subvector(gamma, num_unilaterals, num_bilaterals);
  blaze::DenseSubvector<const DynamicVector<real> > gamma_n = subvector(gamma, 0, num_contacts);

  v_new = M_invk + M_invD * gamma;

#pragma omp parallel for
  for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
    real fric = data_manager->host_data.fric_rigid_rigid[index].x;
    int2 body_id = ids[index];

    real s_v = D_t_T(index * 2 + 0, body_id.x * 6 + 0) * +v_new[body_id.x * 6 + 0] +
               D_t_T(index * 2 + 0, body_id.x * 6 + 1) * +v_new[body_id.x * 6 + 1] +
               D_t_T(index * 2 + 0, body_id.x * 6 + 2) * +v_new[body_id.x * 6 + 2] +
               D_t_T(index * 2 + 0, body_id.x * 6 + 3) * +v_new[body_id.x * 6 + 3] +
               D_t_T(index * 2 + 0, body_id.x * 6 + 4) * +v_new[body_id.x * 6 + 4] +
               D_t_T(index * 2 + 0, body_id.x * 6 + 5) * +v_new[body_id.x * 6 + 5] +

               D_t_T(index * 2 + 0, body_id.y * 6 + 0) * +v_new[body_id.y * 6 + 0] +
               D_t_T(index * 2 + 0, body_id.y * 6 + 1) * +v_new[body_id.y * 6 + 1] +
               D_t_T(index * 2 + 0, body_id.y * 6 + 2) * +v_new[body_id.y * 6 + 2] +
               D_t_T(index * 2 + 0, body_id.y * 6 + 3) * +v_new[body_id.y * 6 + 3] +
               D_t_T(index * 2 + 0, body_id.y * 6 + 4) * +v_new[body_id.y * 6 + 4] +
               D_t_T(index * 2 + 0, body_id.y * 6 + 5) * +v_new[body_id.y * 6 + 5];

    real s_w = D_t_T(index * 2 + 1, body_id.x * 6 + 0) * +v_new[body_id.x * 6 + 0] +
               D_t_T(index * 2 + 1, body_id.x * 6 + 1) * +v_new[body_id.x * 6 + 1] +
               D_t_T(index * 2 + 1, body_id.x * 6 + 2) * +v_new[body_id.x * 6 + 2] +
               D_t_T(index * 2 + 1, body_id.x * 6 + 3) * +v_new[body_id.x * 6 + 3] +
               D_t_T(index * 2 + 1, body_id.x * 6 + 4) * +v_new[body_id.x * 6 + 4] +
               D_t_T(index * 2 + 1, body_id.x * 6 + 5) * +v_new[body_id.x * 6 + 5] +

               D_t_T(index * 2 + 1, body_id.y * 6 + 0) * +v_new[body_id.y * 6 + 0] +
               D_t_T(index * 2 + 1, body_id.y * 6 + 1) * +v_new[body_id.y * 6 + 1] +
               D_t_T(index * 2 + 1, body_id.y * 6 + 2) * +v_new[body_id.y * 6 + 2] +
               D_t_T(index * 2 + 1, body_id.y * 6 + 3) * +v_new[body_id.y * 6 + 3] +
               D_t_T(index * 2 + 1, body_id.y * 6 + 4) * +v_new[body_id.y * 6 + 4] +
               D_t_T(index * 2 + 1, body_id.y * 6 + 5) * +v_new[body_id.y * 6 + 5];

    data_manager->host_data.s[index * 1 + 0] = sqrt(s_v * s_v + s_w * s_w) * fric;
  }
}

void ChConstraintRigidRigid::Build_E() {
  if (data_manager->num_rigid_contacts <= 0) {
    return;
  }
  SOLVERMODE solver_mode = data_manager->settings.solver.solver_mode;
  DynamicVector<real>& E = data_manager->host_data.E;
  uint num_contacts = data_manager->num_rigid_contacts;

#pragma omp parallel for
  for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
    int2 body = data_manager->host_data.bids_rigid_rigid[index];

    real4 cA = data_manager->host_data.compliance_data[body.x];
    real4 cB = data_manager->host_data.compliance_data[body.y];

    real compliance_normal = (cA.x == 0 || cB.x == 0) ? 0 : (cA.x + cB.x) * .5;
    real compliance_sliding = (cA.y == 0 || cB.y == 0) ? 0 : (cA.y + cB.y) * .5;
    real compliance_spinning = (cA.z == 0 || cB.z == 0) ? 0 : (cA.z + cB.z) * .5;
    real compliance_rolling = (cA.w == 0 || cB.w == 0) ? 0 : (cA.w + cB.w) * .5;

    E[index * 1 + 0] = inv_hhpa * compliance_normal;
    if (solver_mode == SLIDING) {
      E[num_contacts + index * 2 + 0] = inv_hhpa * compliance_sliding;
      E[num_contacts + index * 2 + 1] = inv_hhpa * compliance_sliding;
    } else if (solver_mode == SPINNING) {
      E[3 * num_contacts + index * 3 + 0] = inv_hhpa * compliance_spinning;
      E[3 * num_contacts + index * 3 + 1] = inv_hhpa * compliance_rolling;
      E[3 * num_contacts + index * 3 + 2] = inv_hhpa * compliance_rolling;
    }
  }
}

void ChConstraintRigidRigid::Build_D() {
  LOG(INFO) << "ChConstraintRigidRigid::Build_D";
  real3* norm = data_manager->host_data.norm_rigid_rigid.data();
  real3* ptA = data_manager->host_data.cpta_rigid_rigid.data();
  real3* ptB = data_manager->host_data.cptb_rigid_rigid.data();
  real3* pos_data = data_manager->host_data.pos_rigid.data();
  int2* ids = data_manager->host_data.bids_rigid_rigid.data();
  real4* rot = data_manager->host_data.rot_rigid.data();

  SubMatrixType D_n_T = _DNT_;

  const CompressedMatrix<real>& M_inv = data_manager->host_data.M_inv;

  SOLVERMODE solver_mode = data_manager->settings.solver.solver_mode;

  const std::vector<ChBody*>* body_list = data_manager->body_list;

#pragma omp parallel for
  for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
    real3 U = norm[index], V, W;
    real3 T3, T4, T5, T6, T7, T8;
    real3 TA, TB, TC;
    real3 TD, TE, TF;
    Orthogonalize(U, V, W);
    int2 body_id = ids[index];

    int row = index;
    // The position is subtracted here now instead of performing it in the narrowphase
    Compute_Jacobian(rot[body_id.x], U, V, W, ptA[index] - pos_data[body_id.x], T3, T4, T5);
    Compute_Jacobian(rot[body_id.y], U, V, W, ptB[index] - pos_data[body_id.y], T6, T7, T8);

    // Normal jacobian entries
    SetRow6(D_n_T, row * 1 + 0, body_id.x * 6, -U, T3);
    SetRow6(D_n_T, row * 1 + 0, body_id.y * 6, U, -T6);

    if (solver_mode == SLIDING || solver_mode == SPINNING) {
      SubMatrixType D_t_T = _DTT_;
      SetRow6(D_t_T, row * 2 + 0, body_id.x * 6, -V, T4);
      SetRow6(D_t_T, row * 2 + 1, body_id.x * 6, -W, T5);

      SetRow6(D_t_T, row * 2 + 0, body_id.y * 6, V, -T7);
      SetRow6(D_t_T, row * 2 + 1, body_id.y * 6, W, -T8);
    }

    if (solver_mode == SPINNING) {
      SubMatrixType D_s_T = _DST_;
      Compute_Jacobian_Rolling(rot[body_id.x], U, V, W, TA, TB, TC);
      Compute_Jacobian_Rolling(rot[body_id.y], U, V, W, TD, TE, TF);

      SetRow3(D_s_T, row * 3 + 0, body_id.x * 6 + 3, -TA);
      SetRow3(D_s_T, row * 3 + 1, body_id.x * 6 + 3, -TB);
      SetRow3(D_s_T, row * 3 + 2, body_id.x * 6 + 3, -TC);

      SetRow3(D_s_T, row * 3 + 0, body_id.y * 6 + 3, TD);
      SetRow3(D_s_T, row * 3 + 1, body_id.y * 6 + 3, TE);
      SetRow3(D_s_T, row * 3 + 2, body_id.y * 6 + 3, TF);
    }
  }
}

void ChConstraintRigidRigid::GenerateSparsity() {
  LOG(INFO) << "ChConstraintRigidRigid::GenerateSparsity";
  SOLVERMODE solver_mode = data_manager->settings.solver.solver_mode;

  CompressedMatrix<real>& D_n_T = data_manager->host_data.D_T;
  CompressedMatrix<real>& D_t_T = data_manager->host_data.D_T;
  CompressedMatrix<real>& D_s_T = data_manager->host_data.D_T;

  const int2* ids = data_manager->host_data.bids_rigid_rigid.data();

  for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
    int2 body_id = ids[index];
    int row = index;
    int off = 0;

    D_n_T.append(off + row * 1 + 0, body_id.x * 6 + 0, 1);
    D_n_T.append(off + row * 1 + 0, body_id.x * 6 + 1, 1);
    D_n_T.append(off + row * 1 + 0, body_id.x * 6 + 2, 1);

    D_n_T.append(off + row * 1 + 0, body_id.x * 6 + 3, 1);
    D_n_T.append(off + row * 1 + 0, body_id.x * 6 + 4, 1);
    D_n_T.append(off + row * 1 + 0, body_id.x * 6 + 5, 1);

    D_n_T.append(off + row * 1 + 0, body_id.y * 6 + 0, 1);
    D_n_T.append(off + row * 1 + 0, body_id.y * 6 + 1, 1);
    D_n_T.append(off + row * 1 + 0, body_id.y * 6 + 2, 1);

    D_n_T.append(off + row * 1 + 0, body_id.y * 6 + 3, 1);
    D_n_T.append(off + row * 1 + 0, body_id.y * 6 + 4, 1);
    D_n_T.append(off + row * 1 + 0, body_id.y * 6 + 5, 1);

    D_n_T.finalize(off + row * 1 + 0);
  }

  if (solver_mode == SLIDING || solver_mode == SPINNING) {
    for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
      int2 body_id = ids[index];
      int row = index;
      int off = data_manager->num_rigid_contacts;
      D_t_T.append(off + row * 2 + 0, body_id.x * 6 + 0, 1);
      D_t_T.append(off + row * 2 + 0, body_id.x * 6 + 1, 1);
      D_t_T.append(off + row * 2 + 0, body_id.x * 6 + 2, 1);

      D_t_T.append(off + row * 2 + 0, body_id.x * 6 + 3, 1);
      D_t_T.append(off + row * 2 + 0, body_id.x * 6 + 4, 1);
      D_t_T.append(off + row * 2 + 0, body_id.x * 6 + 5, 1);

      D_t_T.append(off + row * 2 + 0, body_id.y * 6 + 0, 1);
      D_t_T.append(off + row * 2 + 0, body_id.y * 6 + 1, 1);
      D_t_T.append(off + row * 2 + 0, body_id.y * 6 + 2, 1);

      D_t_T.append(off + row * 2 + 0, body_id.y * 6 + 3, 1);
      D_t_T.append(off + row * 2 + 0, body_id.y * 6 + 4, 1);
      D_t_T.append(off + row * 2 + 0, body_id.y * 6 + 5, 1);

      D_t_T.finalize(off + row * 2 + 0);

      D_t_T.append(off + row * 2 + 1, body_id.x * 6 + 0, 1);
      D_t_T.append(off + row * 2 + 1, body_id.x * 6 + 1, 1);
      D_t_T.append(off + row * 2 + 1, body_id.x * 6 + 2, 1);

      D_t_T.append(off + row * 2 + 1, body_id.x * 6 + 3, 1);
      D_t_T.append(off + row * 2 + 1, body_id.x * 6 + 4, 1);
      D_t_T.append(off + row * 2 + 1, body_id.x * 6 + 5, 1);

      D_t_T.append(off + row * 2 + 1, body_id.y * 6 + 0, 1);
      D_t_T.append(off + row * 2 + 1, body_id.y * 6 + 1, 1);
      D_t_T.append(off + row * 2 + 1, body_id.y * 6 + 2, 1);

      D_t_T.append(off + row * 2 + 1, body_id.y * 6 + 3, 1);
      D_t_T.append(off + row * 2 + 1, body_id.y * 6 + 4, 1);
      D_t_T.append(off + row * 2 + 1, body_id.y * 6 + 5, 1);

      D_t_T.finalize(off + row * 2 + 1);
    }
  }

  if (solver_mode == SPINNING) {
    for (int index = 0; index < data_manager->num_rigid_contacts; index++) {
      int2 body_id = ids[index];
      int row = index;
      int off = 3 * data_manager->num_rigid_contacts;
      D_s_T.append(off + row * 3 + 0, body_id.x * 6 + 3, 0);
      D_s_T.append(off + row * 3 + 0, body_id.x * 6 + 4, 0);
      D_s_T.append(off + row * 3 + 0, body_id.x * 6 + 5, 0);

      D_s_T.append(off + row * 3 + 0, body_id.y * 6 + 3, 0);
      D_s_T.append(off + row * 3 + 0, body_id.y * 6 + 4, 0);
      D_s_T.append(off + row * 3 + 0, body_id.y * 6 + 5, 0);

      D_s_T.finalize(off + row * 3 + 0);

      D_s_T.append(off + row * 3 + 1, body_id.x * 6 + 3, 0);
      D_s_T.append(off + row * 3 + 1, body_id.x * 6 + 4, 0);
      D_s_T.append(off + row * 3 + 1, body_id.x * 6 + 5, 0);

      D_s_T.append(off + row * 3 + 1, body_id.y * 6 + 3, 0);
      D_s_T.append(off + row * 3 + 1, body_id.y * 6 + 4, 0);
      D_s_T.append(off + row * 3 + 1, body_id.y * 6 + 5, 0);

      D_s_T.finalize(off + row * 3 + 1);

      D_s_T.append(off + row * 3 + 2, body_id.x * 6 + 3, 0);
      D_s_T.append(off + row * 3 + 2, body_id.x * 6 + 4, 0);
      D_s_T.append(off + row * 3 + 2, body_id.x * 6 + 5, 0);

      D_s_T.append(off + row * 3 + 2, body_id.y * 6 + 3, 0);
      D_s_T.append(off + row * 3 + 2, body_id.y * 6 + 4, 0);
      D_s_T.append(off + row * 3 + 2, body_id.y * 6 + 5, 0);

      D_s_T.finalize(off + row * 3 + 2);
    }
  }
}
