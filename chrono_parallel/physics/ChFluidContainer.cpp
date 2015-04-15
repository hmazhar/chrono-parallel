
#include <stdlib.h>
#include <algorithm>

#include "physics/ChSystem.h"
#include "core/ChLinearAlgebra.h"
#include "core/ChMemory.h"  // must be last include (memory leak debugger). In .cpp only.
#include <chrono_parallel/physics/ChFluidContainer.h>
namespace chrono {

using namespace collision;
using namespace geometry;

//////////////////////////////////////
//////////////////////////////////////

/// CLASS FOR A 3DOF FLUID NODE

ChFluidContainer::ChFluidContainer(real r) {
  body_id = 0;
  data_manager = 0;
}

ChFluidContainer::~ChFluidContainer() {
}

ChFluidContainer::ChFluidContainer(const ChFluidContainer& other) : ChPhysicsItem(other) {
  this->body_id = other.body_id;
  this->data_manager = other.data_manager;
}

ChFluidContainer& ChFluidContainer::operator=(const ChFluidContainer& other) {
  if (&other == this)
    return *this;

  ChPhysicsItem::operator=(other);
  return *this;
}

}  // END_OF_NAMESPACE____

/////////////////////
