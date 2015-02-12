#ifndef CHC_NARROWPHASE_H
#define CHC_NARROWPHASE_H


#include "chrono_parallel/ChDataManager.h"
#include "chrono_parallel/ChParallelDefines.h"

namespace chrono {
namespace collision {

struct ConvexShape {
   shape_type type;  //type of shape
   real3 A;  //location
   real3 B;  //dimensions
   real3 C;  //extra
   quaternion R;  //rotation
   real3* convex;      // pointer to convex data;
};

}  // end namespace collision
}  // end namespace chrono

#endif

