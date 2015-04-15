
#include <stdlib.h>
#include <algorithm>

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "core/ChLinearAlgebra.h"
#include "core/ChMemory.h"  // must be last include (memory leak debugger). In .cpp only.
#include <chrono_parallel/physics/ChFluidContainer.h>

namespace chrono {

using namespace collision;
using namespace geometry;

//////////////////////////////////////
//////////////////////////////////////

/// CLASS FOR A 3DOF FLUID NODE

ChFluidContainer::ChFluidContainer(ChSystemParallel* physics_system) {
  system = physics_system;
}

ChFluidContainer::~ChFluidContainer() {
}

ChFluidContainer::ChFluidContainer(const ChFluidContainer& other) : ChPhysicsItem(other) {
  this->system = other.system;
}

ChFluidContainer& ChFluidContainer::operator=(const ChFluidContainer& other) {
  if (&other == this)
    return *this;

  ChPhysicsItem::operator=(other);
  return *this;
}

void ChFluidContainer::AddFluid(const std::vector<real3>& positions, const std::vector<real3>& velocities) {
  host_vector<real3>& pos_fluid = system->data_manager->host_data.pos_fluid;
  host_vector<real3>& vel_fluid = system->data_manager->host_data.vel_fluid;

  pos_fluid.insert(pos_fluid.end(), positions.begin(), positions.end());
  vel_fluid.insert(vel_fluid.end(), velocities.begin(), velocities.end());
  // In case the number of velocities provided were not enough, resize to the number of fluid bodies
  vel_fluid.resize(pos_fluid.size());
  system->data_manager->num_fluid_bodies = pos_fluid.size();
}

real3 ChFluidContainer::GetPos(int i) {
  return system->data_manager->host_data.pos_fluid[i];
}
void ChFluidContainer::SetPos(const int& i, const real3& mpos) {
  system->data_manager->host_data.pos_fluid[i] = mpos;
}

real3 ChFluidContainer::GetPos_dt(int i) {
  return system->data_manager->host_data.vel_fluid[i];
}
void ChFluidContainer::SetPos_dt(const int& i, const real3& mposdt) {
  system->data_manager->host_data.vel_fluid[i] = mposdt;
}

}  // END_OF_NAMESPACE____

/////////////////////
