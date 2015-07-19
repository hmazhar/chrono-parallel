#include "chrono_parallel/solver/ChSolverPGS.h"
#include <blaze/math/SparseRow.h>
#include <blaze/math/CompressedVector.h>
using namespace chrono;

uint ChSolverPGS::SolvePGS(const uint max_iter,
                           const uint size,
                           DynamicVector<real>& mb,
                           DynamicVector<real>& ml) {
    real& residual = data_manager->measures.solver.residual;
    real& objective_value = data_manager->measures.solver.objective_value;
    uint num_contacts = data_manager->num_rigid_contacts;
    CompressedMatrix<real>&  Nshur = data_manager->host_data.Nshur;
    N_gamma_new.resize(size);
    temp.resize(size);

    diagonal.resize(num_contacts, false);
    real g_diff = 1.0 / pow(size, 2.0);
    real omega = data_manager->settings.solver.omega;
    data_manager->system_timer.start("ChSolverParallel_Solve");
    #pragma omp parallel for
    for (size_t i = 0; i < num_contacts; ++i) {
        diagonal[i] = 3.0/(
        		Nshur(i, i) +
				Nshur(num_contacts + i * 2 + 0, num_contacts + i * 2 + 0) +
				Nshur(num_contacts + i * 2 + 1,num_contacts + i * 2 + 1)
				);
      }
    data_manager->system_timer.stop("ChSolverParallel_Solve");

    for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {
    	data_manager->system_timer.start("ChSolverParallel_Solve");
    	for (size_t i = 0; i < num_contacts; ++i) {
          int a = i * 1 + 0;
          int b = num_contacts + i * 2 + 0;
          int c = num_contacts + i * 2 + 1;
          real Dinv = diagonal [i];
          ml[a] = ml[a] - omega * Dinv * ((row(Nshur, a), ml) - mb[a]);
          ml[b] = ml[b] - omega * Dinv * ((row(Nshur, b), ml) - mb[b]);
          ml[c] = ml[c] - omega * Dinv * ((row(Nshur, c), ml) - mb[c]);

        Project_Single(i, ml.data());
      }
      data_manager->system_timer.stop("ChSolverParallel_Solve");

      ShurProduct(ml, N_gamma_new);
      temp = ml - g_diff * (N_gamma_new - mb);
      Project(temp.data());
      temp = (1.0 / g_diff) * (ml - temp);
      real temp_dotb = (real)(temp, temp);
      real residual = sqrt(temp_dotb);


      objective_value = GetObjective(ml, mb);
      AtIterationEnd(residual, objective_value, data_manager->system_timer.GetTime("ChSolverParallel_Solve"));
    }

    return current_iteration;
  return 0;
}
