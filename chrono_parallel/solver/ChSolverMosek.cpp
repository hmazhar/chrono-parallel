#include "chrono_parallel/solver/ChSolverMosek.h"
#include <blaze/math/CompressedVector.h>

using namespace chrono;

ChSolverMosek::ChSolverMosek() : ChSolverParallel() {
  env = NULL;
  task = NULL;

  // Create the Mosek environment
  res_code = MSK_makeenv(&env, NULL);
}

ChSolverMosek::~ChSolverMosek() {
  if (res_code == MSK_RES_OK) {
    MSK_deleteenv(&env);
  }
}
static void MSKAPI printstr(void* handle, MSKCONST char str[]) {
  printf("%s", str);
} /* printstr */

void ConvertCOO(const CompressedMatrix<real>& mat,
                std::vector<MSKint32t>& row,
                std::vector<MSKint32t>& col,
                std::vector<real>& val) {
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

uint ChSolverMosek::SolveMosek(const uint max_iter,
                               const uint size,
                               const blaze::DynamicVector<real>& rhs,
                               blaze::DynamicVector<real>& gamma) {
  const DynamicVector<real>& v = data_manager->host_data.v;

  const CompressedMatrix<real>& M_inv = data_manager->host_data.M_inv;
  uint num_dof = data_manager->num_dof;
  uint num_contacts = data_manager->num_rigid_contacts;
  uint num_bilaterals = data_manager->num_bilaterals;
  uint num_constraints = data_manager->num_constraints;
  uint num_unilaterals = data_manager->num_unilaterals;
  uint nnz_bilaterals = data_manager->nnz_bilaterals;
  uint nnz_unilaterals = 6 * 6 * num_contacts;

  int nnz_total = nnz_unilaterals + nnz_bilaterals;

  // PDIP specific vectors
  blaze::CompressedMatrix<real>& D_T = data_manager->host_data.D_T;
  blaze::CompressedMatrix<real>& M_invD = data_manager->host_data.M_invD;

  CompressedMatrix<real> N(num_constraints, num_constraints);

  // int2* ids = data_manager->host_data.bids_rigid_rigid.data();

  std::vector<MSKint32t> obj_row;
  std::vector<MSKint32t> obj_col;
  std::vector<real> obj_val;

  // CompressedMatrix<real> N = data_manager->host_data.D_n_T * data_manager->host_data.M_invD_n;
  N = D_T * M_invD;

  ConvertCOO(N, obj_row, obj_col, obj_val);

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

  MSKint32t csub[3];
  blaze::DynamicVector<real> rhs_neg = -rhs;

  // Create the optimization task.
  if (res_code == MSK_RES_OK) {
    res_code = MSK_maketask(env, num_constr, num_variables, &task);
  }

  if (res_code == MSK_RES_OK) {
    // Connects a user-defined function to a task stream.
     //res_code = MSK_linkfunctotaskstream(task, MSK_STREAM_LOG, NULL, printstr);

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

    // Let them be FREEEE!
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
  data_manager->system_timer.start("ChSolverParallel_Solve");
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
    MSKrealt primalobj ;

    res_code = MSK_getprimalobj(task, MSK_SOL_ITR, &primalobj);
    std::cout<<primalobj<<std::endl;
  }
  data_manager->system_timer.stop("ChSolverParallel_Solve");
  if (!res_code) {
    // res_code = MSK_solutionsummary(task, MSK_STREAM_MSG);
  }

  // Cleanup
  if (res_code == MSK_RES_OK) {
    res_code = MSK_deletetask(&task);
  }

  // Usually exit when saving model so that it does not get overwritten
  //  exit(1);
  return current_iteration;
}
