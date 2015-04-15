// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Hammad Mazhar
// =============================================================================
// The boradphase algorithm uses a spatial subdivison approach to find contacts
// between objects of different sizes
// =============================================================================

#ifndef CHC_BROADPHASE_H
#define CHC_BROADPHASE_H

#include "chrono_parallel/ChParallelDefines.h"
#include "chrono_parallel/math/ChParallelMath.h"
#include "chrono_parallel/ChDataManager.h"
#include "chrono_parallel/collision/ChCAABBGenerator.h"

namespace chrono {
namespace collision {

class CH_PARALLEL_API ChCBroadphase {
 public:
  // functions
  ChCBroadphase();
  void DetectPossibleCollisions();
  void DetermineBoundingBox();
  void OffsetAABB();
  void ComputeTiledGrid();
  void ComputeOneLevelGrid();
  void OneLevelBroadphase();

  ChParallelDataManager* data_manager;

 private:
  // GENERAL INFORMATION======================================================================================
  // trans_pos_fluid holds the transformed fluid position, used only in the CD
  host_vector<real3> trans_fluid_pos;
  uint num_aabb_rigid;
  uint num_aabb_fluid;
  // TILED GRID INFORMATION===================================================================================
  real tile_size;
  real inv_tile_size;
  int3 tiles_per_axis;
  uint rigid_tiles_active;
  // BINNED GRID INFORMATION==================================================================================
  real3 inv_bin_size;
  uint num_bins_active;
  uint number_of_bin_intersections;
  uint number_of_contacts_possible;

  custom_vector<uint> bins_intersected;
  custom_vector<uint> bin_number;
  custom_vector<uint> aabb_number;
  custom_vector<uint> bin_start_index;
  custom_vector<uint> num_contact;
  // =========================================================================================================
};
}
}

#endif
