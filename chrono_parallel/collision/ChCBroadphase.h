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
  void ProjectRigidOntoTiledGrid();
  void AddFluidToGrid();
  void FlagTiles();
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
  // TILED GRID INFORMATION RIGID=============================================================================
  host_vector<uint> rigid_tiles_intersected;  // The number of tiles intersected by the rigid aabbs
  host_vector<uint> rigid_tile_number;        // The tile numbers which were intersected by rigid aabbs
  uint num_rigid_tile_intersections;          // How many tiles do the rigid aabbs intersect
  uint rigid_tiles_active;                    // The number of unique tiles intersected
  // TILED GRID INFORMATION FLUID=============================================================================
  host_vector<uint> fluid_tile_number;       // The tile number containing the centroid of each fluid, aka active tiles
  host_vector<int> fluid_aabb_number;        // The corresponding fluid num for each fluid_tile_number, -1 is rigid
  host_vector<bool> fluid_flag;              // This flag is 1 if the fluid is not in a cell with a rigid, 0 if it is
  host_vector<uint> fluid_interactions;      // The number of interactions for each fluid particle
  host_vector<uint> fluid_tile_start_index;  //
  uint number_of_fluid_interactions;         // The number of total fluid-fluid contacts
  uint fluid_tiles_active;                   // The number of unique tiles
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
