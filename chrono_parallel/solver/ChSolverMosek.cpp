#include "chrono_parallel/solver/ChSolverMosek.h"
#include <blaze/math/CompressedVector.h>
#include "mosek.h"
using namespace chrono;

ChSolverMosek::ChSolverMosek() : ChSolverParallel() {}

static void MSKAPI printstr(void* handle, MSKCONST char str[]) { printf("%s", str); } /* printstr */

void ConvertCOO(const CompressedMatrix<real>& mat, std::vector<MSKint32t>& row, std::vector<MSKint32t>& col, std::vector<real>& val) {
  int nnz = mat.nonZeros();
  row.resize(nnz);
  col.resize(nnz);
  val.resize(nnz);
  int count = 0;
  for (int i = 0; i < mat.rows(); ++i) {
    for (blaze::CompressedMatrix<double>::Iterator it = mat.begin(i); it != mat.end(i); ++it) {
      if (it->index() <= i) {
        row[count] = i;
        col[count] = it->index();
        val[count] = it->value();
        count++;
      }
    }
  }
}

uint ChSolverMosek::SolveMosek(const uint max_iter, const uint size, const blaze::DynamicVector<real>& rhs, blaze::DynamicVector<real>& gamma) {
  const DynamicVector<real>& v = data_container->host_data.v;

  const CompressedMatrix<real>& M_inv = data_container->host_data.M_inv;
  uint num_dof = data_container->num_dof;
  uint num_contacts = data_container->num_contacts;
  uint num_bilaterals = data_container->num_bilaterals;
  uint num_constraints = data_container->num_constraints;
  uint num_unilaterals = data_container->num_unilaterals;
  uint nnz_bilaterals = data_container->nnz_bilaterals;
  uint nnz_unilaterals = 6 * 6 * data_container->num_contacts;

  int nnz_total = nnz_unilaterals + nnz_bilaterals;

  D_T.reserve(nnz_total);
  D_T.resize(num_constraints, num_dof, false);

  M_invD.reserve(nnz_total);
  M_invD.resize(num_dof, num_constraints, false);

  CompressedMatrix<real> N(num_constraints,num_constraints);

  blaze::submatrix(D_T, 0, 0, num_contacts, num_dof) = data_container->host_data.D_n_T;
  blaze::submatrix(D_T, num_contacts, 0, 2 * num_contacts, num_dof) = data_container->host_data.D_t_T;
  blaze::submatrix(D_T, num_unilaterals, 0, num_bilaterals, num_dof) = data_container->host_data.D_b_T;

//  blaze::submatrix(D, 0, 0, num_dof, num_contacts) = data_container->host_data.D_n;
//  blaze::submatrix(D, 0, num_contacts, num_dof, 2 * num_contacts) = data_container->host_data.D_t;
//  blaze::submatrix(D, 0, num_unilaterals, num_dof, num_bilaterals) = data_container->host_data.D_b;

  blaze::submatrix(M_invD, 0, 0, num_dof, num_contacts) = data_container->host_data.M_invD_n;
  blaze::submatrix(M_invD, 0, num_contacts, num_dof, 2 * num_contacts) = data_container->host_data.M_invD_t;
  blaze::submatrix(M_invD, 0, num_unilaterals, num_dof, num_bilaterals) = data_container->host_data.M_invD_b;

  // int2* ids = data_container->host_data.bids_rigid_rigid.data();

  std::vector<MSKint32t> obj_row;
  std::vector<MSKint32t> obj_col;
  std::vector<real> obj_val;

  // CompressedMatrix<real> N = data_container->host_data.D_n_T * data_container->host_data.M_invD_n;
  N = D_T * M_invD;

  ConvertCOO(N, obj_row, obj_col, obj_val);

  MSKrescodee res_code;    // Variable for holding the error code

  const MSKint32t num_variables = num_constraints;
  const MSKint32t num_constr = num_contacts + num_bilaterals;

  std::vector<MSKboundkey_enum> bound_free(num_variables, MSK_BK_FR);
  std::vector<real> bound_lo(num_variables, -MSK_INFINITY);
  std::vector<real> bound_up(num_variables, +MSK_INFINITY);

  //  std::vector<MSKint32t> cone_size(num_contacts, 3);
  //  std::vector<MSKconetypee> cone_type(num_contacts, MSK_CT_QUAD);
  //  std::vector<MSKrealt> cone_par(num_contacts, 0.0);
  //
  //  std::vector<MSKint32t> cone_index(num_contacts * 3);

  MSKenv_t env = NULL;      // Mosek Environment variable
  MSKtask_t task = NULL;    // Task that mosek will perform
  MSKint32t csub[3];
  blaze::DynamicVector<real> rhs_neg = -rhs;

  // Create the Mosek environment
  res_code = MSK_makeenv(&env, NULL);

  // Create the optimization task.
  if (res_code == MSK_RES_OK) {
    res_code = MSK_maketask(env, num_constr, num_variables, &task);
  }
  if (res_code == MSK_RES_OK) {

    // Connects a user-defined function to a task stream.
     MSK_linkfunctotaskstream(task, MSK_STREAM_LOG, NULL, printstr);

    // Append 'numcon' empty constraints. The constraints will initially have no bounds.
    if (res_code == MSK_RES_OK) {
      res_code = MSK_appendcons(task, num_constr);
    }
    // Append 'numvar' variables. The variables will be fixed at zero initially.
    if (res_code == MSK_RES_OK) {
      res_code = MSK_appendvars(task, num_variables);
    }
    // Add the Q matrix for the quadratic objective
    if (res_code == MSK_RES_OK) {
      res_code = MSK_putqobj(task, N.nonZeros(), obj_row.data(), obj_col.data(), obj_val.data());
    }

    // Let them be FREE!
    if (res_code == MSK_RES_OK) {
      res_code = MSK_putvarboundslice(task, 0, num_variables, bound_free.data(), bound_lo.data(), bound_up.data());
    }

    // Supply the linear term in the objective
    if (res_code == MSK_RES_OK) {
      res_code = MSK_putcslice(task, 0, num_variables, rhs_neg.data());
    }
    // Add all of the conic constraints, note that this cannot be done in parallel
    for (int index = 0; index < num_contacts && res_code == MSK_RES_OK; ++index) {
      csub[0] = index * 1 + 0;
      csub[1] = num_contacts + index * 2 + 0;
      csub[2] = num_contacts + index * 2 + 1;
      res_code = MSK_appendcone(task, MSK_CT_QUAD, 0.0, 3, csub);
    }
  }

  // Model created, lets solve!
  if (res_code == MSK_RES_OK) {
    MSKrescodee trmcode;
    // convert the QCQP to a QCP
    if (res_code == MSK_RES_OK) {
      res_code = MSK_toconic(task);
    }
    // Output the model to a file
    if (res_code == MSK_RES_OK) {
      // res_code = MSK_writedata(task, "taskdump.opf");
    }
    // Run optimizer
    if (res_code == MSK_RES_OK) {
      res_code = MSK_optimizetrm(task, &trmcode);
    }
    // Print a summary containing information about the solution for debugging purposes
    //
    if (res_code == MSK_RES_OK) {
      MSK_getxxslice(task, MSK_SOL_ITR, 0, num_variables, gamma.data());
    }
  }

  if(!res_code){
    res_code = MSK_solutionsummary(task, MSK_STREAM_MSG);
  }

  // Cleanup
  if (res_code == MSK_RES_OK) {
    res_code = MSK_deletetask(&task);
  }
  if (res_code == MSK_RES_OK) {
    MSK_deleteenv(&env);
  }
  // Usually exit when saving model so that it does not get overwritten
  //  exit(1);
  return current_iteration;
}