#include "chrono_parallel/solver/ChSolverJacobi.h"
#include <blaze/math/SparseRow.h>
#include <blaze/math/CompressedVector.h>
using namespace chrono;

uint ChSolverJacobi::SolveJacobi(const uint max_iter, const uint size,
		DynamicVector<real>& mb, DynamicVector<real>& ml) {
	real& residual = data_manager->measures.solver.residual;
	real& objective_value = data_manager->measures.solver.objective_value;

	uint num_contacts = data_manager->num_rigid_contacts;
	diagonal.resize(num_contacts, false);
	ml_old = ml;
	CompressedMatrix<real>& Nshur = data_manager->host_data.Nshur;
	real g_diff = 1.0 / pow(size, 2.0);

	N_gamma_new.resize(size);
	temp.resize(size);
	 data_manager->system_timer.start("ChSolverParallel_Solve");
//	for (size_t i = 0; i < num_contacts; ++i) {
//		diagonal[i * 1 + 0] = Nshur(i, i);
//		diagonal[num_contacts + i * 2 + 0] = Nshur(num_contacts + i * 2 + 0,
//				num_contacts + i * 2 + 0);
//		diagonal[num_contacts + i * 2 + 1] = Nshur(num_contacts + i * 2 + 1,
//				num_contacts + i * 2 + 1);
//	}
#pragma omp parallel for
    for (size_t i = 0; i < num_contacts; ++i) {
        diagonal[i] = 3.0/(
        		Nshur(i, i) +
				Nshur(num_contacts + i * 2 + 0, num_contacts + i * 2 + 0) +
				Nshur(num_contacts + i * 2 + 1,num_contacts + i * 2 + 1)
				);
      }
	 data_manager->system_timer.stop("ChSolverParallel_Solve");
	//Project(ml.data());
	//
	for (current_iteration = 0; current_iteration < max_iter;current_iteration++) {
		 data_manager->system_timer.start("ChSolverParallel_Solve");
		real omega = data_manager->settings.solver.omega;
		N_gamma_old = Nshur * ml_old - mb;
#pragma omp parallel for
		for (int i = 0; i < num_contacts; ++i) {
			int a = i * 1 + 0;
			int b = num_contacts + i * 2 + 0;
			int c = num_contacts + i * 2 + 1;

			//real Dinv = 3.0 / (diagonal[a] + diagonal[b] + diagonal[c]);
			real Dinv = diagonal [i];
			ml[a] = ml[a] - omega * Dinv * (N_gamma_old[a]);
			ml[b] = ml[b] - omega * Dinv * (N_gamma_old[b]);
			ml[c] = ml[c] - omega * Dinv * (N_gamma_old[c]);
		}
		Project(ml.data());
		ml_old = ml;

		 data_manager->system_timer.stop("ChSolverParallel_Solve");

		ShurProduct(ml, N_gamma_new);
		temp = ml - g_diff * (N_gamma_new - mb);
		Project(temp.data());
		temp = (1.0 / g_diff) * (ml - temp);
		real temp_dotb = (real) (temp, temp);
		real residual = sqrt(temp_dotb);
		objective_value =  GetObjective(ml, mb);
		AtIterationEnd(residual, objective_value,data_manager->system_timer.GetTime("ChSolverParallel_Solve"));
	}

	return current_iteration;
	return 0;
}
