#include <algorithm>

// not used but prevents compilation errors with cuda 7 RC
#include <thrust/transform.h>

#include <thrust/iterator/constant_iterator.h>
#include <chrono_parallel/collision/ChCBroadphase.h>
#include "chrono_parallel/collision/ChCBroadphaseUtils.h"

using thrust::transform;
using thrust::transform_reduce;

namespace chrono {
namespace collision {

// Function to Count AABB Bin intersections=================================================================
inline void function_Count_AABB_BIN_Intersection(const uint index,
                                                 const real3& inv_bin_size,
                                                 const host_vector<real3>& aabb_min_data,
                                                 const host_vector<real3>& aabb_max_data,
                                                 host_vector<uint>& bins_intersected) {
  int3 gmin = HashMin(aabb_min_data[index], inv_bin_size);
  int3 gmax = HashMax(aabb_max_data[index], inv_bin_size);
  bins_intersected[index] = (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
}

// Function to Store AABB Bin Intersections=================================================================
inline void function_Store_AABB_BIN_Intersection(const uint index,
                                                 const int3& bins_per_axis,
                                                 const real3& inv_bin_size,
                                                 const host_vector<real3>& aabb_min_data,
                                                 const host_vector<real3>& aabb_max_data,
                                                 const host_vector<uint>& bins_intersected,
                                                 host_vector<uint>& bin_number,
                                                 host_vector<uint>& aabb_number) {
  uint count = 0, i, j, k;
  int3 gmin = HashMin(aabb_min_data[index], inv_bin_size);
  int3 gmax = HashMax(aabb_max_data[index], inv_bin_size);
  uint mInd = bins_intersected[index];
  for (i = gmin.x; i <= gmax.x; i++) {
    for (j = gmin.y; j <= gmax.y; j++) {
      for (k = gmin.z; k <= gmax.z; k++) {
        bin_number[mInd + count] = Hash_Index(I3(i, j, k), bins_per_axis);
        aabb_number[mInd + count] = index;
        count++;
      }
    }
  }
}

// Function to count AABB AABB intersection=================================================================
inline void function_Count_AABB_AABB_Intersection(const uint index,
                                                  const host_vector<real3>& aabb_min_data,
                                                  const host_vector<real3>& aabb_max_data,
                                                  const host_vector<uint>& bin_number,
                                                  const host_vector<uint>& aabb_number,
                                                  const host_vector<uint>& bin_start_index,
                                                  const host_vector<short2>& fam_data,
                                                  const host_vector<bool>& body_active,
                                                  const host_vector<uint>& body_id,
                                                  host_vector<uint>& num_contact) {
  uint start = bin_start_index[index];
  uint end = bin_start_index[index + 1];
  uint count = 0;
  // Terminate early if there is only one object in the bin
  if (end - start == 1) {
    num_contact[index] = 0;
    return;
  }
  for (uint i = start; i < end; i++) {
    uint shapeA = aabb_number[i];
    real3 Amin = aabb_min_data[shapeA];
    real3 Amax = aabb_max_data[shapeA];
    short2 famA = fam_data[shapeA];
    uint bodyA = body_id[shapeA];

    for (uint k = i + 1; k < end; k++) {
      uint shapeB = aabb_number[k];
      uint bodyB = body_id[shapeB];

      if (shapeA == shapeB)
        continue;
      if (bodyA == bodyB)
        continue;
      if (!body_active[bodyA] && !body_active[bodyB])
        continue;
      if (!collide(famA, fam_data[shapeB]))
        continue;
      if (!overlap(Amin, Amax, aabb_min_data[shapeB], aabb_max_data[shapeB]))
        continue;
      count++;
    }
  }

  num_contact[index] = count;
}

// Function to store AABB-AABB intersections================================================================
inline void function_Store_AABB_AABB_Intersection(const uint index,
                                                  const host_vector<real3>& aabb_min_data,
                                                  const host_vector<real3>& aabb_max_data,
                                                  const host_vector<uint>& bin_number,
                                                  const host_vector<uint>& aabb_number,
                                                  const host_vector<uint>& bin_start_index,
                                                  const host_vector<uint>& num_contact,
                                                  const host_vector<short2>& fam_data,
                                                  const host_vector<bool>& body_active,
                                                  const host_vector<uint>& body_id,
                                                  host_vector<long long>& potential_contacts) {
  uint start = bin_start_index[index];
  uint end = bin_start_index[index + 1];
  // Terminate early if there is only one object in the bin
  if (end - start == 1) {
    return;
  }
  uint offset = num_contact[index];
  uint count = 0;

  for (uint i = start; i < end; i++) {
    uint shapeA = aabb_number[i];
    real3 Amin = aabb_min_data[shapeA];
    real3 Amax = aabb_max_data[shapeA];
    short2 famA = fam_data[shapeA];
    uint bodyA = body_id[shapeA];

    for (int k = i + 1; k < end; k++) {
      uint shapeB = aabb_number[k];
      uint bodyB = body_id[shapeB];

      if (shapeA == shapeB)
        continue;
      if (bodyA == bodyB)
        continue;
      if (!body_active[bodyA] && !body_active[bodyB])
        continue;
      if (!collide(famA, fam_data[shapeB]))
        continue;
      if (!overlap(Amin, Amax, aabb_min_data[shapeB], aabb_max_data[shapeB]))
        continue;

      if (shapeB < shapeA) {
        uint t = shapeA;
        shapeA = shapeB;
        shapeB = t;
      }
      // the two indices of the shapes that make up the contact
      potential_contacts[offset + count] = ((long long)shapeA << 32 | (long long)shapeB);
      count++;
    }
  }
}
// =========================================================================================================

void function_Check_Neighbors(const int& current_number,
                              const uint& tile_number,
                              const real3& current_pos,
                              const real kernel_radius,
                              const host_vector<uint>& fluid_tile_start_index,
                              const host_vector<int>& fluid_aabb_number,
                              const host_vector<real3>& trans_fluid_pos,
                              uint& count,
                              uint& count_rigid) {
  uint start = fluid_tile_start_index[tile_number];
  uint end = fluid_tile_start_index[tile_number + 1];

  std::cout << "function_Check_Neighbors " << start << " " << end << std::endl;
  for (uint i = start; i < end; i++) {
    int neighbor_number = fluid_aabb_number[i];
    std::cout << "neighbor_number " << neighbor_number << " " << current_number << std::endl;
    if (current_number == neighbor_number) {
      continue;  // Skip the same body
    }
    if (neighbor_number < 0) {
      count_rigid++;
      continue;
    }
    // otherwise it is a fluid particle
    real3 neighbor_pos = trans_fluid_pos[neighbor_number];
    if (function_Check_Sphere(current_pos, neighbor_pos, kernel_radius)) {
      count++;
    }
  }
}
// =========================================================================================================

void function_Store_Neighbors(const int& current_number,
                              const uint& tile_number,
                              const real3& current_pos,
                              const real kernel_radius,
                              const uint& offset,
                              const uint& offset_rigid,
                              const host_vector<uint>& fluid_tile_start_index,
                              const host_vector<int>& fluid_aabb_number,
                              const host_vector<real3>& trans_fluid_pos,
                              uint& count,
                              uint& count_rigid,
                              host_vector<int2>& bids_fluid_fluid,
                              host_vector<long long>& bids_rigid_fluid) {
  uint start = fluid_tile_start_index[tile_number];
  uint end = fluid_tile_start_index[tile_number + 1];
  for (uint i = start; i < end; i++) {
    int neighbor_number = fluid_aabb_number[i];
    if (current_number == neighbor_number || neighbor_number == -1) {
      continue;  // Skip the same body and boudnary ones
    }
    if (neighbor_number < 0) {
      bids_rigid_fluid[offset_rigid + count_rigid] = ((long long)current_number << 32 | (long long)neighbor_number);
      count_rigid++;
      continue;
    }
    real3 neighbor_pos = trans_fluid_pos[neighbor_number];
    if (function_Check_Sphere(current_pos, neighbor_pos, kernel_radius)) {
      bids_fluid_fluid[offset + count] = I2(current_number, neighbor_number);
      count++;
    }
  }
}

// =========================================================================================================
ChCBroadphase::ChCBroadphase() {
  number_of_contacts_possible = 0;
  num_bins_active = 0;
  number_of_bin_intersections = 0;
  data_manager = 0;
}

// DETERMINE BOUNDS ========================================================================================

void ChCBroadphase::DetermineBoundingBox() {
  bbox_transformation unary_op;
  bbox_reduction binary_op;

  host_vector<real3>& aabb_min_rigid = data_manager->host_data.aabb_min_rigid;
  host_vector<real3>& aabb_max_rigid = data_manager->host_data.aabb_max_rigid;
  host_vector<real3>& aabb_min_fluid = data_manager->host_data.aabb_min_fluid;
  host_vector<real3>& aabb_max_fluid = data_manager->host_data.aabb_max_fluid;
  // Determine the bounds on the total space, create a zero volume bounding box using the first aabb
  bbox res(aabb_min_rigid[0], aabb_min_rigid[0]);
  res = transform_reduce(aabb_min_rigid.begin(), aabb_min_rigid.end(), unary_op, res, binary_op);
  res = transform_reduce(aabb_max_rigid.begin(), aabb_max_rigid.end(), unary_op, res, binary_op);
  res = transform_reduce(aabb_min_fluid.begin(), aabb_min_fluid.end(), unary_op, res, binary_op);
  res = transform_reduce(aabb_max_fluid.begin(), aabb_max_fluid.end(), unary_op, res, binary_op);

  data_manager->measures.collision.min_bounding_point = res.first;
  data_manager->measures.collision.max_bounding_point = res.second;
  data_manager->measures.collision.global_origin = res.first;

  thrust::constant_iterator<real3> offset(res.first);

  LOG(TRACE) << "Minimum bounding point: (" << res.first.x << ", " << res.first.y << ", " << res.first.z << ")";
  LOG(TRACE) << "Maximum bounding point: (" << res.second.x << ", " << res.second.y << ", " << res.second.z << ")";
}
// OFFSET AABB =============================================================================================
void ChCBroadphase::OffsetAABB() {
  host_vector<real3>& aabb_min_rigid = data_manager->host_data.aabb_min_rigid;
  host_vector<real3>& aabb_max_rigid = data_manager->host_data.aabb_max_rigid;
  host_vector<real3>& aabb_min_fluid = data_manager->host_data.aabb_min_fluid;
  host_vector<real3>& aabb_max_fluid = data_manager->host_data.aabb_max_fluid;
  thrust::constant_iterator<real3> offset(data_manager->measures.collision.global_origin);
  transform(aabb_min_rigid.begin(), aabb_min_rigid.end(), offset, aabb_min_rigid.begin(), thrust::minus<real3>());
  transform(aabb_max_rigid.begin(), aabb_max_rigid.end(), offset, aabb_max_rigid.begin(), thrust::minus<real3>());
  transform(aabb_min_fluid.begin(), aabb_min_fluid.end(), offset, aabb_min_fluid.begin(), thrust::minus<real3>());
  transform(aabb_max_fluid.begin(), aabb_max_fluid.end(), offset, aabb_max_fluid.begin(), thrust::minus<real3>());
  host_vector<real3>& pos_fluid = data_manager->host_data.pos_fluid;
  trans_fluid_pos.resize(pos_fluid.size());
  transform(pos_fluid.begin(), pos_fluid.end(), offset, trans_fluid_pos.begin(), thrust::minus<real3>());
}
// COMPUTE TILED GRID=======================================================================================
void ChCBroadphase::ComputeTiledGrid() {
  if (num_aabb_fluid == 0) {
    //  return;
  }
  const real3& min_bounding_point = data_manager->measures.collision.min_bounding_point;
  const real3& max_bounding_point = data_manager->measures.collision.max_bounding_point;
  tile_size = data_manager->settings.fluid.kernel_radius * 2;

  real3 diagonal = (fabs(max_bounding_point - min_bounding_point));
  tiles_per_axis = I3(diagonal / R3(tile_size));

  tiles_per_axis.x += 1;
  tiles_per_axis.y += 1;
  tiles_per_axis.z += 1;

  inv_tile_size = 1.0 / tile_size;

  LOG(TRACE) << "tile_size: " << tile_size;
  LOG(TRACE) << "tiles_per_axis: (" << tiles_per_axis.x << ", " << tiles_per_axis.y << ", " << tiles_per_axis.z << ")";
}

// =========================================================================================================
void ChCBroadphase::ComputeOneLevelGrid() {
  const real3& min_bounding_point = data_manager->measures.collision.min_bounding_point;
  const real3& max_bounding_point = data_manager->measures.collision.max_bounding_point;
  int3& bins_per_axis = data_manager->settings.collision.bins_per_axis;
  real3& bin_size_vec = data_manager->measures.collision.bin_size_vec;
  const real density = data_manager->settings.collision.grid_density;

  real3 diagonal = (fabs(max_bounding_point - min_bounding_point));
  int num_shapes = num_aabb_rigid;

  if (data_manager->settings.collision.fixed_bins == false) {
    bins_per_axis = function_Compute_Grid_Resolution(num_shapes, diagonal, density);
  }
  bin_size_vec = diagonal / R3(bins_per_axis.x, bins_per_axis.y, bins_per_axis.z);
  inv_bin_size = 1.0 / bin_size_vec;
  LOG(TRACE) << "bin_size_vec: (" << bin_size_vec.x << ", " << bin_size_vec.y << ", " << bin_size_vec.z << ")";
}

void ChCBroadphase::ProjectRigidOntoTiledGrid() {
  if (num_aabb_fluid == 0) {
    //  return;
  }
  host_vector<real3>& aabb_min_rigid = data_manager->host_data.aabb_min_rigid;
  host_vector<real3>& aabb_max_rigid = data_manager->host_data.aabb_max_rigid;

  rigid_tiles_intersected.resize(num_aabb_rigid + 1);
  rigid_tiles_intersected[num_aabb_rigid] = 0;

#pragma omp parallel for
  for (int index = 0; index < num_aabb_rigid; index++) {
    int3 gmin = HashMin(aabb_min_rigid[index], inv_tile_size);
    int3 gmax = HashMax(aabb_max_rigid[index], inv_tile_size);
    rigid_tiles_intersected[index] = (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
  }
  Thrust_Exclusive_Scan(rigid_tiles_intersected);
  num_rigid_tile_intersections = rigid_tiles_intersected.back();
  LOG(TRACE) << "num_rigid_tile_intersections: " << num_rigid_tile_intersections;
  rigid_tile_number.resize(num_rigid_tile_intersections);
  rigid_tile_aabb.resize(num_rigid_tile_intersections);

#pragma omp parallel for
  for (int index = 0; index < num_aabb_rigid; index++) {
    uint count = 0, i, j, k;
    int3 gmin = HashMin(aabb_min_rigid[index], inv_tile_size);
    int3 gmax = HashMax(aabb_max_rigid[index], inv_tile_size);
    uint mInd = rigid_tiles_intersected[index];
    for (i = gmin.x; i <= gmax.x; i++) {
      for (j = gmin.y; j <= gmax.y; j++) {
        for (k = gmin.z; k <= gmax.z; k++) {
          rigid_tile_number[mInd + count] = Hash_Index(I3(i, j, k), tiles_per_axis);
          rigid_tile_aabb[mInd + count] = -index - 1;
          count++;
        }
      }
    }
  }
  // The AABB number is NOT important here we only want to know the bins (maybe)
  // Thrust_Sort(rigid_tile_number);
  // Thrust_Sort_By_Key(rigid_tile_number, rigid_tile_aabb);
  // Thrust_Unique();
  // rigid_tiles_active = Thrust_Reduce_By_Key(rigid_tile_number, rigid_tile_number, rigid_tile_start_index);

  // rigid_tile_start_index.resize(rigid_tiles_active + 1);
  // rigid_tile_start_index[rigid_tiles_active] = 0;
  // Thrust_Exclusive_Scan(rigid_tile_start_index);

  // LOG(TRACE) << "rigid_tiles_active: " << rigid_tiles_active;
  // At the end of this step we have a list of all of the tiles that have rigid bodies
}
// DETERMINE TILE NUMBERS FOR FLUID=========================================================================
void ChCBroadphase::AddFluidToGrid() {
  fluid_tile_number.resize(num_aabb_fluid);
#pragma omp parallel for
  for (int i = 0; i < num_aabb_fluid; i++) {
    fluid_tile_number[i] = Hash_Index(HashMin(trans_fluid_pos[i], inv_tile_size), tiles_per_axis);
  }
}

// =========================================================================================================
void ChCBroadphase::FlagTiles() {
  // Add the rigid tiles to the fluid ones
  LOG(TRACE) << "fluid_tile_number size: " << num_aabb_fluid + num_rigid_tile_intersections;
  fluid_tile_number.resize(num_aabb_fluid + num_rigid_tile_intersections);
  // Copy the flagged tile numbers into the list
  thrust::copy(rigid_tile_number.begin(), rigid_tile_number.end(), fluid_tile_number.begin() + num_aabb_fluid);
  // Add -1 for the tiles flagged with rigid bodies, if a tile has a -1 in it it means that
  // a rigid body is in this tile
  fluid_aabb_number.resize(num_aabb_fluid + num_rigid_tile_intersections);
  // Fill fluid_aabb_number with a sequence representing the fluid particle numbers
  thrust::sequence(fluid_aabb_number.begin(), fluid_aabb_number.begin() + num_aabb_fluid, 0);
  // Fill the extra AABB indices with -1, this means that a rigid body is in this tile
  thrust::copy(rigid_tile_aabb.begin(), rigid_tile_aabb.end(), fluid_aabb_number.begin() + num_aabb_fluid);
  // thrust::fill(fluid_aabb_number.begin() + num_aabb_fluid, fluid_aabb_number.end(), -1);

  Thrust_Sort_By_Key(fluid_tile_number, fluid_aabb_number);
  LOG(TRACE) << "fluid_tile_number: ";
  //  for (int i = 0; i < fluid_tile_number.size(); i++) {
  //    std::cout << fluid_tile_number[i] << " " << fluid_aabb_number[i] << std::endl;
  //  }

  fluid_tile_start_index.resize(num_aabb_fluid + num_rigid_tile_intersections);
  fluid_tiles_active = Thrust_Reduce_By_Key(fluid_tile_number, fluid_tile_number, fluid_tile_start_index);

  fluid_tile_start_index.resize(fluid_tiles_active + 1);
  fluid_tile_start_index[fluid_tiles_active] = 0;

  Thrust_Exclusive_Scan(fluid_tile_start_index);
  LOG(TRACE) << "fluid_tile_start_index: ";
  //  for (int i = 0; i < fluid_tile_start_index.size(); i++) {
  //    std::cout << fluid_tile_number[i] << " " << fluid_tile_start_index[i] << std::endl;
  //  }
  tile_active.resize(tiles_per_axis.x * tiles_per_axis.y * tiles_per_axis.z);
  Thrust_Fill(tile_active, -1);
  for (int i = 0; i < fluid_tiles_active; i++) {
    tile_active[fluid_tile_number[i]] = i;
  }
  //  LOG(TRACE) << "tile_active: ";

  //  for (int i = 0; i < tile_active.size(); i++) {
  //    std::cout << i << " " << tile_active[i] << std::endl;
  //  }

  LOG(TRACE) << "fluid_tiles_active: " << fluid_tiles_active;
}

// =========================================================================================================

void ChCBroadphase::FluidContacts() {
  fluid_flag.resize(num_aabb_fluid);
  thrust::fill(fluid_flag.begin(), fluid_flag.end(), 1);

  fluid_interactions.resize(num_aabb_fluid + 1);
  fluid_interactions[num_aabb_fluid] = 0;
  rigid_interactions.resize(num_aabb_fluid + 1);
  rigid_interactions[num_aabb_fluid] = 0;
  const real radius = data_manager->settings.fluid.kernel_radius;
// Count contacts for each fluid particle
// fluid_aabb_number is longer than num_aabb_fluid and contains extra -1's
#pragma omp parallel for
  for (int index = 0; index < num_aabb_fluid; index++) {
    real3 current_pos = trans_fluid_pos[index];
    int3 tile_position = HashMin(current_pos, inv_tile_size);
    uint tile_number = Hash_Index(tile_position, tiles_per_axis);
    uint count = 0, count_rigid = 0;
    // Our grid does NOT wrap, so clamp it
    for (int a = tile_position.x - 1; a <= tile_position.x + 1; a++) {
      if (a < 0 || a >= tiles_per_axis.x) {
        continue;
      }
      for (int b = tile_position.y - 1; b <= tile_position.y + 1; b++) {
        if (b < 0 || b >= tiles_per_axis.y) {
          continue;
        }
        for (int c = tile_position.z - 1; c <= tile_position.z + 1; c++) {
          if (c < 0 || c >= tiles_per_axis.z) {
            continue;
          }

          uint tile_number = Hash_Index(I3(a, b, c), tiles_per_axis);
          if (tile_active[tile_number] == -1) {
            continue;
          }

          uint start = fluid_tile_start_index[tile_active[tile_number]];
          uint end = fluid_tile_start_index[tile_active[tile_number] + 1];

          for (uint i = start; i < end; i++) {
            int neighbor_number = fluid_aabb_number[i];
            if (index == neighbor_number) {
              continue;  // Skip the same body or any lesser ones
            }
            if (neighbor_number < 0) {
              count_rigid++;
              continue;
            }
            if (index >= neighbor_number) {
              continue;  // Skip the same body or any lesser ones
            }
            // otherwise it is a fluid particle
            real3 neighbor_pos = trans_fluid_pos[neighbor_number];
            if (function_Check_Sphere(current_pos, neighbor_pos, radius)) {
              count++;
            }
          }
        }
      }
    }

    fluid_interactions[index] = count;
    rigid_interactions[index] = count_rigid;
  }

  Thrust_Exclusive_Scan(fluid_interactions);
  Thrust_Exclusive_Scan(rigid_interactions);
  number_of_fluid_interactions = fluid_interactions.back();
  number_of_rigid_interactions = rigid_interactions.back();
  LOG(TRACE) << "number_of_fluid_interactions " << number_of_fluid_interactions;
  LOG(TRACE) << "number_of_rigid_interactions " << number_of_rigid_interactions;
  data_manager->host_data.bids_fluid_fluid.resize(number_of_fluid_interactions);
  data_manager->host_data.pair_rigid_fluid.resize(number_of_rigid_interactions);

#pragma omp parallel for
  for (int index = 0; index < num_aabb_fluid; index++) {
    real3 current_pos = trans_fluid_pos[index];
    int3 tile_position = HashMin(current_pos, inv_tile_size);
    uint tile_number = Hash_Index(tile_position, tiles_per_axis);
    uint offset = fluid_interactions[index];
    uint offset_rigid = rigid_interactions[index];
    uint count = 0, count_rigid = 0;

    for (int a = tile_position.x - 1; a <= tile_position.x + 1; a++) {
      if (a < 0 || a >= tiles_per_axis.x) {
        continue;
      }
      for (int b = tile_position.y - 1; b <= tile_position.y + 1; b++) {
        if (b < 0 || b >= tiles_per_axis.y) {
          continue;
        }
        for (int c = tile_position.z - 1; c <= tile_position.z + 1; c++) {
          if (c < 0 || c >= tiles_per_axis.z) {
            continue;
          }

          uint tile_number = Hash_Index(I3(a, b, c), tiles_per_axis);

          if (tile_active[tile_number] == -1) {
            continue;
          }

          uint start = fluid_tile_start_index[tile_active[tile_number]];
          uint end = fluid_tile_start_index[tile_active[tile_number] + 1];
          for (uint i = start; i < end; i++) {
            int neighbor_number = fluid_aabb_number[i];
            if (index == neighbor_number) {
              continue;  // Skip the same body and boudnary ones
            }
            if (neighbor_number < 0) {
              data_manager->host_data.pair_rigid_fluid[offset_rigid + count_rigid] =
                  ((long long)index + num_aabb_fluid << 32 | (long long)((neighbor_number + 1) * -1));
              count_rigid++;
              continue;
            }
            if (index >= neighbor_number) {
              continue;  // Skip the same body and boudnary ones
            }
            real3 neighbor_pos = trans_fluid_pos[neighbor_number];
            if (function_Check_Sphere(current_pos, neighbor_pos, radius)) {
              data_manager->host_data.bids_fluid_fluid[offset + count] = I2(index, neighbor_number);
              count++;
            }
          }
        }
      }
    }
  }

  thrust::stable_sort(data_manager->host_data.pair_rigid_fluid.begin(), data_manager->host_data.pair_rigid_fluid.end());
  number_of_rigid_interactions = Thrust_Unique(data_manager->host_data.pair_rigid_fluid);

  LOG(TRACE) << "number_of_rigid_interactions " << number_of_rigid_interactions;
  //
  //  host_vector<real3>& aabb_min_fluid = data_manager->host_data.aabb_min_fluid;
  //  host_vector<real3>& aabb_max_fluid = data_manager->host_data.aabb_max_fluid;

  // Sort the fluid by the flag and resize the aabb list
  // the fluid_aabb_number vector holds the actual number of the fluid particle (needed in narrowphase)

  // auto zip_start = thrust::make_zip_iterator(
  //    thrust::make_tuple(fluid_aabb_number.begin(), aabb_min_fluid.begin(), aabb_max_fluid.begin()));

  // thrust::sort_by_key(fluid_flag.begin(), fluid_flag.end(), zip_start);
  // resize everything to the flagged fluid particles
  // fluid_aabb_number.resize(num_fluid_flagged);
  // aabb_min_fluid.resize(num_fluid_flagged);
  // aabb_max_fluid.resize(num_fluid_flagged);

  // uint num_rigid_bodies = data_manager->num_rigid_bodies;
  // thrust::constant_iterator<uint> offset(num_rigid_bodies);
  // Need to make sure that fluid number does not clash with object number, offset by the number of rigid_bodies
  // transform(fluid_aabb_number.begin(), fluid_aabb_number.end(), offset, fluid_aabb_number.begin(),
  // thrust::plus<int>());
}

void ChCBroadphase::CheckRigidFluidPairs() {
  host_vector<bool> rigid_fluid_pairs(number_of_rigid_interactions);
  Thrust_Fill(rigid_fluid_pairs, 1);
#pragma omp parallel for
  for (int i = 0; i < number_of_rigid_interactions; i++) {
    long long pair = data_manager->host_data.pair_rigid_fluid[i];
    int2 pair2 = I2(int(pair >> 32), int(pair & 0xffffffff));

    // get the aabbs

    real3 aabb_min_fluid = aabb_min_fluid[pair2.x - num_aabb_fluid];
    real3 aabb_max_fluid = aabb_max_fluid[pair2.x - num_aabb_fluid];

    real3 aabb_min_rigid = aabb_min_rigid[pair2.y];
    real3 aabb_max_rigid = aabb_max_rigid[pair2.y];

    if (overlap(aabb_min_fluid, aabb_max_fluid, aabb_min_rigid, aabb_max_rigid)) {
      rigid_fluid_pairs[i] = 0;
    }
  }
  Thrust_Sort_By_Key(rigid_fluid_pairs, data_manager->host_data.pair_rigid_fluid);

  uint count_active = Thrust_Count(rigid_fluid_pairs, 0);

  rigid_fluid_pairs.resize(count_active);
}

// =========================================================================================================
// use spatial subdivision to detect the list of POSSIBLE collisions
// let user define their own narrow-phase collision detection

void ChCBroadphase::OneLevelBroadphase() {
  host_vector<real3>& aabb_min_rigid = data_manager->host_data.aabb_min_rigid;
  host_vector<real3>& aabb_max_rigid = data_manager->host_data.aabb_max_rigid;
  host_vector<real3>& aabb_min_fluid = data_manager->host_data.aabb_min_fluid;
  host_vector<real3>& aabb_max_fluid = data_manager->host_data.aabb_max_fluid;

  host_vector<long long>& contact_pairs = data_manager->host_data.pair_rigid_rigid;
  int3& bins_per_axis = data_manager->settings.collision.bins_per_axis;
  const host_vector<short2>& fam_data = data_manager->host_data.fam_rigid;
  const host_vector<bool>& obj_active = data_manager->host_data.active_rigid;
  const host_vector<uint>& obj_data_id = data_manager->host_data.id_rigid;

  uint num_shapes = num_aabb_rigid;

  bins_intersected.resize(num_shapes + 1);
  bins_intersected[num_shapes] = 0;

#pragma omp parallel for
  for (int i = 0; i < num_shapes; i++) {
    function_Count_AABB_BIN_Intersection(i, inv_bin_size, aabb_min_rigid, aabb_max_rigid, bins_intersected);
  }

  Thrust_Exclusive_Scan(bins_intersected);
  number_of_bin_intersections = bins_intersected.back();

  LOG(TRACE) << "Number of bin intersections: " << number_of_bin_intersections;

  bin_number.resize(number_of_bin_intersections);
  aabb_number.resize(number_of_bin_intersections);
  bin_start_index.resize(number_of_bin_intersections);

#pragma omp parallel for
  for (int i = 0; i < num_shapes; i++) {
    function_Store_AABB_BIN_Intersection(i, bins_per_axis, inv_bin_size, aabb_min_rigid, aabb_max_rigid,
                                         bins_intersected, bin_number, aabb_number);
  }

  LOG(TRACE) << "Completed (device_Store_AABB_BIN_Intersection)";

  Thrust_Sort_By_Key(bin_number, aabb_number);
  num_bins_active = Thrust_Reduce_By_Key(bin_number, bin_number, bin_start_index);

  if (num_bins_active <= 0) {
    number_of_contacts_possible = 0;
    return;
  }

  bin_start_index.resize(num_bins_active + 1);
  bin_start_index[num_bins_active] = 0;

  LOG(TRACE) << bins_per_axis.x << " " << bins_per_axis.y << " " << bins_per_axis.z;
  LOG(TRACE) << "Last active bin: " << num_bins_active;

  Thrust_Exclusive_Scan(bin_start_index);
  num_contact.resize(num_bins_active + 1);
  num_contact[num_bins_active] = 0;

#pragma omp parallel for
  for (int i = 0; i < num_bins_active; i++) {
    function_Count_AABB_AABB_Intersection(i, aabb_min_rigid, aabb_max_rigid, bin_number, aabb_number, bin_start_index,
                                          fam_data, obj_active, obj_data_id, num_contact);
  }

  thrust::exclusive_scan(num_contact.begin(), num_contact.end(), num_contact.begin());
  number_of_contacts_possible = num_contact.back();
  contact_pairs.resize(number_of_contacts_possible);
  LOG(TRACE) << "Number of possible collisions: " << number_of_contacts_possible;

#pragma omp parallel for
  for (int index = 0; index < num_bins_active; index++) {
    function_Store_AABB_AABB_Intersection(index, aabb_min_rigid, aabb_max_rigid, bin_number, aabb_number,
                                          bin_start_index, num_contact, fam_data, obj_active, obj_data_id,
                                          contact_pairs);
  }

  thrust::stable_sort(thrust_parallel, contact_pairs.begin(), contact_pairs.end());
  number_of_contacts_possible = Thrust_Unique(contact_pairs);
  contact_pairs.resize(number_of_contacts_possible);
  LOG(TRACE) << "Number of possible collisions: " << number_of_contacts_possible;
  return;
}

void ChCBroadphase::DetectPossibleCollisions() {
  num_aabb_rigid = data_manager->num_rigid_shapes;
  num_aabb_fluid = data_manager->num_fluid_bodies;
  LOG(TRACE) << "Number of AABBs: " << num_aabb_rigid << ", " << num_aabb_fluid;
  DetermineBoundingBox();
  OffsetAABB();
  ComputeTiledGrid();
  ProjectRigidOntoTiledGrid();
  AddFluidToGrid();
  FlagTiles();
  FluidContacts();
  CheckRigidFluidPairs();
  ComputeOneLevelGrid();
  OneLevelBroadphase();
}
}
}
