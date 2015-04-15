#ifndef CHCONSTRAINT_RIGIDFLUID_H
#define CHCONSTRAINT_RIGIDFLUID_H

#include "chrono_parallel/ChDataManager.h"
#include "chrono_parallel/math/ChParallelMath.h"
namespace chrono {
class CH_PARALLEL_API ChConstraintRigidFluid {
 public:
  ChConstraintRigidFluid() { data_manager = 0; }
  ~ChConstraintRigidFluid() {}
  void Setup(ChParallelDataManager* data_container_) { data_manager = data_container_; }
  void Build_D(SOLVERMODE solver_mode);
  void Build_b(SOLVERMODE solver_mode);
  void Build_E(SOLVERMODE solver_mode);
  void Project(real* gamma);
  void GenerateSparsity(SOLVERMODE solver_mode);

 protected:
  // Pointer to the system's data manager
  ChParallelDataManager* data_manager;
};
}

#endif
