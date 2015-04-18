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

// Count the number of intersections between AABBs and the bins=============================================
void function_Count_AABB_Bin_Intersection(const uint index,
                                          const real3& inv_bin_size,
                                          const host_vector<real3>& aabb_min,
                                          const host_vector<real3>& aabb_max,
                                          uint* bins_intersected) {
  int3 gmin = HashMin(aabb_min[index], inv_bin_size);
  int3 gmax = HashMax(aabb_max[index], inv_bin_size);
  bins_intersected[index] = (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
}
// Store the intersections between AABBs and bins===========================================================
void function_Store_AABB_Bin_Intersection(const uint index,
                                          const real3& inv_bin_size,
                                          const int3& bins_per_axis,
                                          const host_vector<real3>& aabb_min,
                                          const host_vector<real3>& aabb_max,
                                          const host_vector<uint>& bins_intersected,
                                          host_vector<uint>& bin_number,
                                          host_vector<uint>& shape_number) {
  int count = 0, i, j, k;
  int3 gmin = HashMin(aabb_min[index], inv_bin_size);
  int3 gmax = HashMax(aabb_max[index], inv_bin_size);
  uint mInd = bins_intersected[index];
  for (i = gmin.x; i <= gmax.x; i++) {
    for (j = gmin.y; j <= gmax.y; j++) {
      for (k = gmin.z; k <= gmax.z; k++) {
        bin_number[mInd + count] = Hash_Index(I3(i, j, k), bins_per_axis);
        shape_number[mInd + count] = index;
        count++;
      }
    }
  }
}
// For each bin determine the grid size and store it========================================================
void function_Count_Leaves(const uint index,
                           const real density,
                           const real3& bin_size_vec,
                           const uint* bin_start_index,
                           uint* leaves_per_bin) {
  uint start = bin_start_index[index];
  uint end = bin_start_index[index + 1];
  uint num_aabb_in_cell = end - start;

  int3 cell_res = function_Compute_Grid_Resolution(num_aabb_in_cell, bin_size_vec, density);

  leaves_per_bin[index] = cell_res.x * cell_res.y * cell_res.z;
}
// Count the number of AABB leaf intersections for each bin=================================================
void function_Count_AABB_Leaf_Intersection(const uint index,
                                           const real density,
                                           const real3& bin_size_vec,
                                           const int3& bins_per_axis,
                                           const host_vector<uint>& bin_start_index,
                                           const host_vector<uint>& bin_number,
                                           const host_vector<uint>& shape_number,
                                           const host_vector<real3>& aabb_min,
                                           const host_vector<real3>& aabb_max,
                                           host_vector<uint>& leaves_intersected) {
  uint start = bin_start_index[index];
  uint end = bin_start_index[index + 1];
  uint count = 0;
  uint num_aabb_in_cell = end - start;
  int3 cell_res = function_Compute_Grid_Resolution(num_aabb_in_cell, bin_size_vec, density);
  real3 inv_leaf_size = R3(cell_res.x, cell_res.y, cell_res.z) / bin_size_vec;
  int3 bin_index = Hash_Decode(bin_number[index], bins_per_axis);
  real3 bin_position = R3(bin_index.x * bin_size_vec.x, bin_index.y * bin_size_vec.y, bin_index.z * bin_size_vec.z);

  for (uint i = start; i < end; i++) {
    uint shape = shape_number[i];
    // subtract the bin position from the AABB position
    real3 Amin = aabb_min[shape] - bin_position;
    real3 Amax = aabb_max[shape] - bin_position;

    // Make sure that even with subtraction we are at the origin
    Amin = clamp(Amin, R3(0), Amax);

    // Find the extents
    int3 gmin = HashMin(Amin, inv_leaf_size);
    int3 gmax = HashMax(Amax, inv_leaf_size);
    // Make sure that the maximum bin value does not exceed the bounds of this grid
    int3 max_clamp = cell_res - I3(1);
    gmin = clamp(gmin, I3(0), max_clamp);
    gmax = clamp(gmax, I3(0), max_clamp);

    count += (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
  }

  leaves_intersected[index] = count;
}
// Store the AABB leaf intersections for each bin===========================================================
void function_Write_AABB_Leaf_Intersection(const uint& index,
                                           const real density,
                                           const real3& bin_size_vec,
                                           const int3& bin_resolution,
                                           const host_vector<uint>& bin_start_index,
                                           const host_vector<uint>& bin_number,
                                           const host_vector<uint>& bin_shape_number,
                                           const host_vector<real3>& aabb_min,
                                           const host_vector<real3>& aabb_max,
                                           const host_vector<uint>& leaves_intersected,
                                           const host_vector<uint>& leaves_per_bin,
                                           host_vector<uint>& leaf_number,
                                           host_vector<uint>& leaf_shape_number) {
  uint start = bin_start_index[index];
  uint end = bin_start_index[index + 1];
  uint mInd = leaves_intersected[index];
  uint count = 0;
  uint num_aabb_in_cell = end - start;
  int3 cell_res = function_Compute_Grid_Resolution(num_aabb_in_cell, bin_size_vec, density);
  real3 inv_leaf_size = R3(cell_res.x, cell_res.y, cell_res.z) / bin_size_vec;

  int3 bin_index = Hash_Decode(bin_number[index], bin_resolution);

  real3 bin_position = R3(bin_index.x * bin_size_vec.x, bin_index.y * bin_size_vec.y, bin_index.z * bin_size_vec.z);

  for (uint i = start; i < end; i++) {
    uint shape = bin_shape_number[i];
    // subtract the bin position from the AABB position
    real3 Amin = aabb_min[shape] - bin_position;
    real3 Amax = aabb_max[shape] - bin_position;

    // Make sure that even with subtraction we are at the origin
    Amin = clamp(Amin, R3(0), Amax);

    // Find the extents
    int3 gmin = HashMin(Amin, inv_leaf_size);
    int3 gmax = HashMax(Amax, inv_leaf_size);

    // Make sure that the maximum bin value does not exceed the bounds of this grid
    int3 max_clamp = cell_res - I3(1);
    gmin = clamp(gmin, I3(0), max_clamp);
    gmax = clamp(gmax, I3(0), max_clamp);

    int a, b, c;
    for (a = gmin.x; a <= gmax.x; a++) {
      for (b = gmin.y; b <= gmax.y; b++) {
        for (c = gmin.z; c <= gmax.z; c++) {
          leaf_number[mInd + count] = leaves_per_bin[index] + Hash_Index(I3(a, b, c), cell_res);
          leaf_shape_number[mInd + count] = shape;
          count++;
        }
      }
    }
  }
}
// Count the number of AABB AABB intersections for each leaf================================================
void function_Count_AABB_AABB_Intersection(const uint& index,
                                           const host_vector<real3>& aabb_min,
                                           const host_vector<real3>& aabb_max,
                                           const host_vector<uint>& leaf_number,
                                           const host_vector<uint>& leaf_shape_number,
                                           const host_vector<uint>& leaf_start_index,
                                           const host_vector<short2>& fam_data,
                                           const host_vector<bool>& body_active,
                                           const host_vector<uint>& body_id,
                                           host_vector<uint>& num_contact) {
  uint start = leaf_start_index[index];
  uint end = leaf_start_index[index + 1];
  uint count = 0;

  if (end - start == 1) {
    num_contact[index] = 0;
    return;
  }
  for (uint i = start; i < end; i++) {
    uint shapeA = leaf_shape_number[i];
    real3 Amin = aabb_min[shapeA];
    real3 Amax = aabb_max[shapeA];
    short2 famA = fam_data[shapeA];
    uint bodyA = body_id[shapeA];

    for (uint k = i + 1; k < end; k++) {
      uint shapeB = leaf_shape_number[k];
      uint bodyB = body_id[shapeB];
      if (shapeA == shapeB)
        continue;
      if (bodyA == bodyB)
        continue;
      if (!body_active[bodyA] && !body_active[bodyB])
        continue;
      if (!collide(famA, fam_data[shapeB]))
        continue;
      if (!overlap(Amin, Amax, aabb_min[shapeB], aabb_max[shapeB]))
        continue;

      count++;
    }
  }

  num_contact[index] = count;
}
// Store the AABB AABB intersections for each leaf in each bin===============================================
inline void function_Store_AABB_AABB_Intersection(const uint index,
                                                  const host_vector<real3>& aabb_min,
                                                  const host_vector<real3>& aabb_max,
                                                  const uint* leaf_number,
                                                  const uint* leaf_shape_number,
                                                  const uint* leaf_start_index,
                                                  const uint* num_contact,
                                                  const short2* fam_data,
                                                  const bool* body_active,
                                                  const uint* body_id,
                                                  long long* potential_contacts) {
  uint start = leaf_start_index[index];
  uint end = leaf_start_index[index + 1];

  if (end - start == 1)
    return;

  uint Bin = leaf_number[index];
  uint offset = num_contact[index];
  uint count = 0;

  for (uint i = start; i < end; i++) {
    uint shapeA = leaf_shape_number[i];
    real3 Amin = aabb_min[shapeA];
    real3 Amax = aabb_max[shapeA];
    short2 famA = fam_data[shapeA];
    uint bodyA = body_id[shapeA];

    for (int k = i + 1; k < end; k++) {
      uint shapeB = leaf_shape_number[k];
      uint bodyB = body_id[shapeB];

      if (shapeA == shapeB)
        continue;
      if (bodyA == bodyB)
        continue;
      if (!body_active[bodyA] && !body_active[bodyB])
        continue;
      if (!collide(famA, fam_data[shapeB]))
        continue;
      if (!overlap(Amin, Amax, aabb_min[shapeB], aabb_max[shapeB]))
        continue;

      if (shapeB < shapeA) {
        uint t = shapeA;
        shapeA = shapeB;
        shapeB = t;
      }
      potential_contacts[offset + count] =
          ((long long)shapeA << 32 | (long long)shapeB);  // the two indicies of the shapes that make up the contact
      count++;
    }
  }
}
// Determine the bounding box for the objects===============================================================

void ChCBroadphase::DetermineBoundingBox() {
  host_vector<real3>& aabb_min = data_manager->host_data.aabb_min;
  host_vector<real3>& aabb_max = data_manager->host_data.aabb_max;
  // determine the bounds on the total space and subdivide based on the bins per axis
  bbox res(aabb_min[0], aabb_min[0]);
  bbox_transformation unary_op;
  bbox_reduction binary_op;
  res = thrust::transform_reduce(aabb_min.begin(), aabb_min.end(), unary_op, res, binary_op);
  res = thrust::transform_reduce(aabb_max.begin(), aabb_max.end(), unary_op, res, binary_op);
  data_manager->measures.collision.min_bounding_point = res.first;
  data_manager->measures.collision.max_bounding_point = res.second;
  data_manager->measures.collision.global_origin = res.first;

  LOG(TRACE) << "Minimum bounding point: (" << res.first.x << ", " << res.first.y << ", " << res.first.z << ")";
  LOG(TRACE) << "Maximum bounding point: (" << res.second.x << ", " << res.second.y << ", " << res.second.z << ")";
}

void ChCBroadphase::OffsetAABB() {
  host_vector<real3>& aabb_min = data_manager->host_data.aabb_min;
  host_vector<real3>& aabb_max = data_manager->host_data.aabb_max;
  thrust::constant_iterator<real3> offset(data_manager->measures.collision.global_origin);
  thrust::transform(aabb_min.begin(), aabb_min.end(), offset, aabb_min.begin(), thrust::minus<real3>());
  thrust::transform(aabb_max.begin(), aabb_max.end(), offset, aabb_max.begin(), thrust::minus<real3>());
}

// Determine resolution of the top level grid
void ChCBroadphase::ComputeTopLevelResolution() {
  const real3& min_bounding_point = data_manager->measures.collision.min_bounding_point;
  const real3& max_bounding_point = data_manager->measures.collision.max_bounding_point;
  real3& bin_size_vec = data_manager->measures.collision.bin_size_vec;
  const real3& global_origin = data_manager->measures.collision.global_origin;
  const real density = data_manager->settings.collision.grid_density;

  int3& bins_per_axis = data_manager->settings.collision.bins_per_axis;

  // This is the extents of the space aka diameter
  real3 diagonal = (fabs(max_bounding_point - global_origin));
  // Compute the number of slices in this grid level
  if (data_manager->settings.collision.fixed_bins == false) {
    bins_per_axis = function_Compute_Grid_Resolution(num_shapes, diagonal, density);
  }
  bin_size_vec = diagonal / R3(bins_per_axis.x, bins_per_axis.y, bins_per_axis.z);
  LOG(TRACE) << "bin_size_vec: (" << bin_size_vec.x << ", " << bin_size_vec.y << ", " << bin_size_vec.z << ")";

  // Store the inverse for use later
  inv_bin_size = 1.0 / bin_size_vec;
}
// =========================================================================================================
ChCBroadphase::ChCBroadphase() {
  num_shapes = 0;
  number_of_contacts_possible = 0;
  num_active_bins = 0;
  number_of_bin_intersections = 0;
  number_of_leaf_intersections = 0;
  num_active_leaves = 0;
}
// =========================================================================================================
// use spatial subdivision to detect the list of POSSIBLE collisions
// let user define their own narrow-phase collision detection
void ChCBroadphase::DetectPossibleCollisions() {
  const host_vector<real3>& aabb_min = data_manager->host_data.aabb_min;
  const host_vector<real3>& aabb_max = data_manager->host_data.aabb_max;

  num_shapes = data_manager->num_rigid_shapes + data_manager->num_fluid_bodies;

  LOG(TRACE) << "Number of AABBs: " << num_shapes;

  DetermineBoundingBox();
  OffsetAABB();
  ComputeTopLevelResolution();

  int3 bins_per_axis = data_manager->settings.collision.bins_per_axis;
  real3 bin_size_vec = data_manager->measures.collision.bin_size_vec;
  // =========================================================================================================

  bins_intersected.resize(num_shapes + 1);
  bins_intersected[num_shapes] = 0;

// Determine AABB to top level bin count
#pragma omp parallel for
  for (int i = 0; i < num_shapes; i++) {
    function_Count_AABB_Bin_Intersection(i, inv_bin_size, aabb_min, aabb_max, bins_intersected.data());
  }
  Thrust_Exclusive_Scan(bins_intersected);

  number_of_bin_intersections = bins_intersected.back();
  LOG(TRACE) << "number_of_bin_intersections: " << number_of_bin_intersections;
  // Allocate our AABB bin pairs==============================================================================
  bin_number.resize(number_of_bin_intersections);
  bin_aabb_number.resize(number_of_bin_intersections);
  bin_start_index.resize(number_of_bin_intersections);

// Store the bin intersections================================================================================
#pragma omp parallel for
  for (int i = 0; i < num_shapes; i++) {
    function_Store_AABB_Bin_Intersection(i, inv_bin_size, bins_per_axis, aabb_min, aabb_max, bins_intersected,
                                         bin_number, bin_aabb_number);
  }

  // Get sorted top level intersections=======================================================================
  Thrust_Sort_By_Key(bin_number, bin_aabb_number);
  // Number of Top level Intersections========================================================================
  num_active_bins = Thrust_Reduce_By_Key(bin_number, bin_number, bin_start_index);
  LOG(TRACE) << "num_active_bins: " << num_active_bins;
  // Extract Bin ranges=======================================================================================
  bin_start_index.resize(num_active_bins + 1);
  bin_start_index[num_active_bins] = 0;

  Thrust_Exclusive_Scan(bin_start_index);
  // Allocate space to hold leaf cell counts for each bin=====================================================
  leaves_per_bin.resize(num_active_bins + 1);
  leaves_per_bin[num_active_bins] = 0;
// Count leaves in each bin===================================================================================
#pragma omp parallel for
  for (int i = 0; i < num_active_bins; i++) {
    function_Count_Leaves(i, data_manager->settings.collision.leaf_density, bin_size_vec, bin_start_index.data(),
                          leaves_per_bin.data());
  }

  Thrust_Exclusive_Scan(leaves_per_bin);
  // Count leaf intersections=================================================================================
  leaves_intersected.resize(num_active_bins + 1);
  leaves_intersected[num_active_bins] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_active_bins; i++) {
    function_Count_AABB_Leaf_Intersection(i, data_manager->settings.collision.leaf_density, bin_size_vec,
                                          bins_per_axis, bin_start_index, bin_number, bin_aabb_number, aabb_min,
                                          aabb_max, leaves_intersected);
  }

  Thrust_Exclusive_Scan(leaves_intersected);

  number_of_leaf_intersections = leaves_intersected.back();
  LOG(TRACE) << "number_of_leaf_intersections: " << number_of_leaf_intersections;

  leaf_number.resize(number_of_leaf_intersections);
  leaf_aabb_number.resize(number_of_leaf_intersections);
  leaf_start_index.resize(number_of_leaf_intersections);
#pragma omp parallel for
  for (int i = 0; i < num_active_bins; i++) {
    function_Write_AABB_Leaf_Intersection(i, data_manager->settings.collision.leaf_density, bin_size_vec,
                                          bins_per_axis, bin_start_index, bin_number, bin_aabb_number, aabb_min,
                                          aabb_max, leaves_intersected, leaves_per_bin, leaf_number, leaf_aabb_number);
  }
  Thrust_Sort_By_Key(leaf_number, leaf_aabb_number);
  // Number of Leaf Intersections=============================================================================
  num_active_leaves = Thrust_Reduce_By_Key(leaf_number, leaf_number, leaf_start_index);
  LOG(TRACE) << "num_active_leaves: " << num_active_leaves;

  // Extract leaf ranges======================================================================================
  leaf_start_index.resize(num_active_leaves + 1);
  leaf_start_index[num_active_leaves] = 0;

  Thrust_Exclusive_Scan(leaf_start_index);

  const custom_vector<short2>& fam_data = data_manager->host_data.fam_rigid;
  const custom_vector<bool>& obj_active = data_manager->host_data.active_rigid;
  const custom_vector<uint>& obj_data_ID = data_manager->host_data.id_rigid;

  num_contact.resize(num_active_leaves + 1);
  num_contact[num_active_leaves] = 0;

#pragma omp parallel for
  for (int i = 0; i < num_active_leaves; i++) {
    function_Count_AABB_AABB_Intersection(i, aabb_min, aabb_max, leaf_number, leaf_aabb_number, leaf_start_index,
                                          fam_data, obj_active, obj_data_ID, num_contact);
  }
  Thrust_Exclusive_Scan(num_contact);

  custom_vector<long long>& contact_pairs = data_manager->host_data.pair_rigid_rigid;

  number_of_contacts_possible = num_contact.back();
  LOG(TRACE) << "number_of_contacts_possible: " << number_of_contacts_possible;
  contact_pairs.resize(number_of_contacts_possible);
  if (number_of_contacts_possible <= 0) {
    return;
  }

#pragma omp parallel for
  for (int i = 0; i < num_active_leaves; i++) {
    function_Store_AABB_AABB_Intersection(i, aabb_min, aabb_max, leaf_number.data(), leaf_aabb_number.data(),
                                          leaf_start_index.data(), num_contact.data(), fam_data.data(),
                                          obj_active.data(), obj_data_ID.data(), contact_pairs.data());
  }

  thrust::stable_sort(thrust_parallel, contact_pairs.begin(), contact_pairs.end());
  number_of_contacts_possible = thrust::unique(contact_pairs.begin(), contact_pairs.end()) - contact_pairs.begin();

  contact_pairs.resize(number_of_contacts_possible);

  return;
}
}
}
