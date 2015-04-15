// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Hammad Mazhar
// =============================================================================
//
// Description: This file defines a fluid node, it is based on a 3DOF NodeXYZ
// element. This class is similar to ChMatterSPH but is meant to be a bit more
// general.
// =============================================================================

#ifndef CHNODEFLUID_H
#define CHNODEFLUID_H

#include <math.h>

#include "chrono_parallel/ChParallelDefines.h"
#include "chrono_parallel/math/real.h"
#include "chrono_parallel/ChDataManager.h"
namespace chrono {

// Forward references (for parent hierarchy pointer)
class ChSystemParallel;

class CH_PARALLEL_API ChFluidContainer : public ChPhysicsItem {
 public:
  ChFluidContainer(ChSystemParallel* system);
  ~ChFluidContainer();

  ChFluidContainer(const ChFluidContainer& other);             // Copy constructor
  ChFluidContainer& operator=(const ChFluidContainer& other);  // Assignment operator

  void AddFluid(const std::vector<real3>& positions, const std::vector<real3>& velocities);
  void Update(double ChTime);
  void UpdatePosition(double ChTime);
  // Position of the node - in absolute csys.
  real3 GetPos(int i);
  // Position of the node - in absolute csys.
  void SetPos(const int& i, const real3& mpos);

  // Velocity of the node - in absolute csys.
  real3 GetPos_dt(int i);
  // Velocity of the node - in absolute csys.
  void SetPos_dt(const int& i, const real3& mposdt);

 private:
  ChSystemParallel* system;
};
}
#endif
