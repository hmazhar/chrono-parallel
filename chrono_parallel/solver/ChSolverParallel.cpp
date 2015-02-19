#include "chrono_parallel/solver/ChSolverParallel.h"

using namespace chrono;

ChSolverParallel::ChSolverParallel() {
  max_iteration = 100;
  current_iteration = 0;
  rigid_rigid = NULL;
  bilateral = NULL;
}

void ChSolverParallel::Project(real* gamma) {
  data_container->system_timer.start("ChSolverParallel_Project");
  rigid_rigid->Project(gamma);
  data_container->system_timer.stop("ChSolverParallel_Project");
}

void ChSolverParallel::Project_Single(int index, real* gamma) {
  data_container->system_timer.start("ChSolverParallel_Project");
  rigid_rigid->Project_Single(index, gamma);
  data_container->system_timer.stop("ChSolverParallel_Project");
}
//=================================================================================================================================

void ChSolverParallel::ComputeSRhs(custom_vector<real>& gamma,
                                   const custom_vector<real>& rhs,
                                   custom_vector<real3>& vel_data,
                                   custom_vector<real3>& omg_data,
                                   custom_vector<real>& b) {
  // TODO change SHRS to use blaze
  // ComputeImpulses(gamma, vel_data, omg_data);
  // rigid_rigid->ComputeS(rhs, vel_data, omg_data, b);
}

void ChSolverParallel::ShurProduct(const blaze::DynamicVector<real>& x, blaze::DynamicVector<real>& output) {
  const CompressedMatrix<real>& D_n_T = data_container->host_data.D_n_T;
  const CompressedMatrix<real>& D_t_T = data_container->host_data.D_t_T;
  const CompressedMatrix<real>& D_s_T = data_container->host_data.D_s_T;
  const CompressedMatrix<real>& D_b_T = data_container->host_data.D_b_T;

  const CompressedMatrix<real>& M_invD_n = data_container->host_data.M_invD_n;
  const CompressedMatrix<real>& M_invD_t = data_container->host_data.M_invD_t;
  const CompressedMatrix<real>& M_invD_s = data_container->host_data.M_invD_s;
  const CompressedMatrix<real>& M_invD_b = data_container->host_data.M_invD_b;

  const DynamicVector<real>& E = data_container->host_data.E;

  data_container->system_timer.start("ShurProduct");

  uint num_contacts = data_container->num_contacts;
  uint num_unilaterals = data_container->num_unilaterals;
  uint num_bilaterals = data_container->num_bilaterals;

  blaze::DenseSubvector<DynamicVector<real> > o_b = blaze::subvector(output, num_unilaterals, num_bilaterals);
  blaze::DenseSubvector<const DynamicVector<real> > x_b = blaze::subvector(x, num_unilaterals, num_bilaterals);
  blaze::DenseSubvector<const DynamicVector<real> > E_b = blaze::subvector(E, num_unilaterals, num_bilaterals);

  blaze::DenseSubvector<DynamicVector<real> > o_n = blaze::subvector(output, 0, num_contacts);
  blaze::DenseSubvector<const DynamicVector<real> > x_n = blaze::subvector(x, 0, num_contacts);
  blaze::DenseSubvector<const DynamicVector<real> > E_n = blaze::subvector(E, 0, num_contacts);

  switch (data_container->settings.solver.local_solver_mode) {
    case BILATERAL: {
      o_b = D_b_T * (M_invD_b * x_b) + E_b * x_b;
    } break;

    case NORMAL: {
      o_b = D_b_T * (M_invD_b * x_b) + E_b * x_b;
      o_n = D_n_T * (M_invD_n * x_n) + E_n * x_n;
    } break;

    case SLIDING: {
      blaze::DenseSubvector<DynamicVector<real> > o_t = blaze::subvector(output, num_contacts, num_contacts * 2);
      blaze::DenseSubvector<const DynamicVector<real> > x_t = blaze::subvector(x, num_contacts, num_contacts * 2);
      blaze::DenseSubvector<const DynamicVector<real> > E_t = blaze::subvector(E, num_contacts, num_contacts * 2);

      o_b = D_b_T * (M_invD_b * x_b) + E_b * x_b;
      o_n = D_n_T * (M_invD_n * x_n) + E_n * x_n;
      o_t = D_t_T * (M_invD_t * x_t) + E_t * x_t;
    } break;

    case SPINNING: {
      blaze::DenseSubvector<DynamicVector<real> > o_t = blaze::subvector(output, num_contacts, num_contacts * 2);
      blaze::DenseSubvector<const DynamicVector<real> > x_t = blaze::subvector(x, num_contacts, num_contacts * 2);
      blaze::DenseSubvector<const DynamicVector<real> > E_t = blaze::subvector(E, num_contacts, num_contacts * 2);

      blaze::DenseSubvector<DynamicVector<real> > o_s = blaze::subvector(output, num_contacts * 3, num_contacts * 3);
      blaze::DenseSubvector<const DynamicVector<real> > x_s = blaze::subvector(x, num_contacts * 3, num_contacts * 3);
      blaze::DenseSubvector<const DynamicVector<real> > E_s = blaze::subvector(E, num_contacts * 3, num_contacts * 3);
      o_b = D_b_T * (M_invD_b * x_b) + E_b * x_b;
      o_n = D_n_T * (M_invD_n * x_n) + E_n * x_n;
      o_t = D_t_T * (M_invD_t * x_t) + E_t * x_t;
      o_s = D_s_T * (M_invD_s * x_s) + E_s * x_s;
    } break;
  }

  data_container->system_timer.stop("ShurProduct");
}

void ChSolverParallel::ShurBilaterals(const blaze::DynamicVector<real>& x, blaze::DynamicVector<real>& output) {
  const CompressedMatrix<real>& D_b_T = data_container->host_data.D_b_T;
  const CompressedMatrix<real>& M_invD_b = data_container->host_data.M_invD_b;

  output = D_b_T * (M_invD_b * x);
}

//=================================================================================================================================

void ChSolverParallel::UpdatePosition(custom_vector<real>& x) {
  //
  //   if (rigid_rigid->solve_sliding == true || rigid_rigid->solve_spinning ==
  //   true) {
  //      return;
  //   }
  //   shurA(x.data());
  //
  //   data_container->host_data.vel_new_data =
  //   data_container->host_data.vel_data + data_container->host_data.QXYZ_data;
  //   data_container->host_data.omg_new_data +
  //   data_container->host_data.omg_data + data_container->host_data.QUVW_data;
  //
  //#pragma omp parallel for
  //   for (int i = 0; i < data_container->num_bodies; i++) {
  //
  //      data_container->host_data.pos_new_data[i] =
  //      data_container->host_data.pos_data[i] +
  //      data_container->host_data.vel_new_data[i] * step_size;
  //      //real3 dp =
  //      data_container->host_data.pos_new_data[i]-data_container->host_data.pos_data[i];
  //      //cout<<dp<<endl;
  //      real4 moldrot = data_container->host_data.rot_data[i];
  //      real3 newwel = data_container->host_data.omg_new_data[i];
  //
  //      M33 A = AMat(moldrot);
  //      real3 newwel_abs = MatMult(A, newwel);
  //      real mangle = length(newwel_abs) * step_size;
  //      newwel_abs = normalize(newwel_abs);
  //      real4 mdeltarot = Q_from_AngAxis(mangle, newwel_abs);
  //      real4 mnewrot = mdeltarot % moldrot;
  //      data_container->host_data.rot_new_data[i] = mnewrot;
  //   }
}

void ChSolverParallel::UpdateContacts() {
  ////TODO: Re-implement this using new dispatch
  //   if (rigid_rigid->solve_sliding == true || rigid_rigid->solve_spinning ==
  //   true) {
  //      return;
  //   }
  //
  //   //// TODO:  This ASSUMES that we are using an MPR narrowphase!!
  //   ////        Instead of constructing a narrowphaseMPR object here,
  //   ////        modify so that we can use the CHCNarrowphase object
  //   ////        from the CollisionSystemParallel.
  //   collision::ChCNarrowphaseMPR narrowphase;
  //   narrowphase.SetCollisionEnvelope(data_container->settings.collision.collision_envelope);
  //   narrowphase.Update(data_container);
  //
  //   rigid_rigid->UpdateJacobians();
  //   rigid_rigid->UpdateRHS();
}

uint ChSolverParallel::SolveStab(const uint max_iter,
                                 const uint size,
                                 const blaze::DenseSubvector<const DynamicVector<real> >& mb,
                                 blaze::DenseSubvector<DynamicVector<real> >& x) {
  real& residual = data_container->measures.solver.residual;
  custom_vector<real>& iter_hist = data_container->measures.solver.iter_hist;

  uint N = mb.size();

  blaze::DynamicVector<real> v(N, 0), v_hat(x.size()), w(N, 0), w_old, xMR, v_old, Av(x.size()), w_oold;
  real beta, c = 1, eta, norm_rMR, norm_r0, c_old = 1, s_old = 0, s = 0, alpha, beta_old, c_oold, s_oold, r1_hat, r1,
             r2, r3;
  ShurBilaterals(x, v_hat);
  v_hat = mb - v_hat;
  beta = sqrt((v_hat, v_hat));
  w_old = w;
  eta = beta;
  xMR = x;
  norm_rMR = beta;
  norm_r0 = beta;

  if (beta == 0 || norm_rMR / norm_r0 < data_container->settings.solver.tol_speed) {
    return 0;
  }

  for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {
    //// Lanczos
    v_old = v;
    v = 1.0 / beta * v_hat;
    ShurBilaterals(v, Av);
    alpha = (v, Av);
    v_hat = Av - alpha * v - beta * v_old;
    beta_old = beta;
    beta = sqrt((v_hat, v_hat));
    //// QR factorization
    c_oold = c_old;
    c_old = c;
    s_oold = s_old;
    s_old = s;
    r1_hat = c_old * alpha - c_oold * s_old * beta_old;
    r1 = 1 / sqrt(r1_hat * r1_hat + beta * beta);
    r2 = s_old * alpha + c_oold * c_old * beta_old;
    r3 = s_oold * beta_old;
    //// Givens Rotation
    c = r1_hat * r1;
    s = beta * r1;
    //// update
    w_oold = w_old;
    w_old = w;
    w = r1 * (v - r3 * w_oold - r2 * w_old);
    x = x + c * eta * w;
    norm_rMR = norm_rMR * std::abs(s);
    eta = -s * eta;
    residual = norm_rMR / norm_r0;

    real maxdeltalambda = 0;  // CompRes(mb, num_contacts);      //NormInf(ms);
    AtIterationEnd(residual, maxdeltalambda);

    if (residual < data_container->settings.solver.tol_speed) {
      break;
    }
  }
  return current_iteration;
}
