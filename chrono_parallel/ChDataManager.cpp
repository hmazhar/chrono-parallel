#include "chrono_parallel/ChDataManager.h"

using namespace chrono;

ChParallelDataManager::ChParallelDataManager() :
		num_contacts(0),
		num_models(0),
		num_bodies(0),
		num_unilaterals(0),
		num_bilaterals(0),
		num_constraints(0),
		num_shafts(0),
		erad_is_set(false) {}

ChParallelDataManager::~ChParallelDataManager() {}

int ChParallelDataManager::OutputBlazeVector(blaze::DynamicVector<real> src, std::string filename) {

  const char* numformat = "%.16g";
  ChStreamOutAsciiFile stream(filename.c_str());
  stream.SetNumFormat(numformat);

  for (int i = 0; i < src.size(); i++)
    stream << src[i] << "\n";

  return 0;
}

int ChParallelDataManager::OutputBlazeMatrix(blaze::CompressedMatrix<real> src, std::string filename) {

  const char* numformat = "%.16g";
  ChStreamOutAsciiFile stream(filename.c_str());
  stream.SetNumFormat(numformat);

  stream << src.rows() << " " << src.columns() << "\n";
  for (int i = 0; i < src.rows(); ++i) {
    for (CompressedMatrix<real>::Iterator it = src.begin(i); it != src.end(i); ++it) {
      stream << i << " " << it->index() << " " << it->value() << "\n";
    }
  }

  return 0;
}

int ChParallelDataManager::ExportCurrentSystem(std::string output_dir) {
  int offset = 0;
  if (settings.solver.solver_mode == NORMAL) {
    offset = num_contacts;
  } else if (settings.solver.solver_mode == SLIDING) {
    offset = 3 * num_contacts;
  } else if (settings.solver.solver_mode == SPINNING) {
    offset = 6 * num_contacts;
  }

  // fill in the vector for r (rhs vector)
  blaze::DynamicVector<real> r;
  r.resize(num_constraints);
  for (int i = 0; i < r.size(); i++) {
    r[i] = host_data.rhs_data[i];
  }

  // fill in b vector
  blaze::DynamicVector<real> b(num_constraints, 0.0);
  for (int i = 0; i < num_contacts; i++) {
    if (settings.solver.solver_mode == NORMAL) {
      b[i] = host_data.dpth_rigid_rigid[i] / settings.step_size;
    } else if (settings.solver.solver_mode == SLIDING) {
      b[3 * i] = host_data.dpth_rigid_rigid[i] / settings.step_size;
      b[3 * i + 1] = 0.0;
      b[3 * i + 2] = 0.0;
    } else if (settings.solver.solver_mode == SPINNING) {
      b[6 * i] = host_data.dpth_rigid_rigid[i] / settings.step_size;
      b[6 * i + 1] = 0.0;
      b[6 * i + 2] = 0.0;
      b[6 * i + 3] = 0.0;
      b[6 * i + 4] = 0.0;
      b[6 * i + 5] = 0.0;
    }
  }
  for (int i = 0; i < host_data.correction_bilateral.size(); i++) {
    b[i + offset] = host_data.residual_bilateral[i];
  }

  // fill in the information for constraints and friction
  blaze::DynamicVector<real> fric(num_constraints, -2.0);
  for (int i = 0; i < num_contacts; i++) {
    if (settings.solver.solver_mode == NORMAL) {
      fric[i] = host_data.fric_rigid_rigid[i].x;
    } else if (settings.solver.solver_mode == SLIDING) {
      fric[3 * i] = host_data.fric_rigid_rigid[i].x;
      fric[3 * i + 1] = -1;
      fric[3 * i + 2] = -1;
    } else if (settings.solver.solver_mode == SPINNING) {
      fric[6 * i] = host_data.fric_rigid_rigid[i].x;
      fric[6 * i + 1] = -1;
      fric[6 * i + 2] = -1;
      fric[6 * i + 3] = -1;
      fric[6 * i + 4] = -1;
      fric[6 * i + 5] = -1;
    }
  }

  // output r
  std::string filename = output_dir + "dump_r.dat";
  OutputBlazeVector(r, filename);

  // output b
  filename = output_dir + "dump_b.dat";
  OutputBlazeVector(b, filename);

  // output friction data
  filename = output_dir + "dump_fric.dat";
  OutputBlazeVector(fric, filename);

  // output D_T
  filename = output_dir + "dump_D.dat";
  OutputBlazeMatrix(host_data.D_T, filename);

  // output M_inv
  filename = output_dir + "dump_Minv.dat";
  OutputBlazeMatrix(host_data.M_inv, filename);

  return 0;
}
