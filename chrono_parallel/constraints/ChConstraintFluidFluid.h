#ifndef CHCONSTRAINT_FLUIDFLUID_H
#define CHCONSTRAINT_FLUIDFLUID_H

#include "chrono_parallel/ChDataManager.h"
#include "chrono_parallel/math/ChParallelMath.h"

namespace chrono {
class CH_PARALLEL_API ChConstraintFluidFluid {
 public:
  ChConstraintFluidFluid() { data_manager = 0; }
  ~ChConstraintFluidFluid() {}
  void Setup(ChParallelDataManager* data_container_) { data_manager = data_container_; }
  void Build_D();
  void Build_b();
  void Build_E();
  // Based on the list of contacts figure out what each fluid particles
  // neighbors are
  void DetermineNeighbors();
  void Project(real* gamma);
  void GenerateSparsity();

 protected:
  //  custom_vector<int> fluid_contact_idA;
  //  custom_vector<int> fluid_contact_idB, fluid_start_index;
  //  custom_vector<long long> fluid_contact_pair;
  //  custom_vector<real> contact_density;
  //  custom_vector<int> reduced_index_A;
  //  custom_vector<real3> off_diagonal;
  //  int last_body;
  //
  //  custom_vector<M33> shear_tensor;
  // Pointer to the system's data manager
  ChParallelDataManager* data_manager;
};
}

#endif
