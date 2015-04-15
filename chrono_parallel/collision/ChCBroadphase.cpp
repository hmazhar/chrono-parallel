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
  const real3& min_bounding_point = data_manager->measures.collision.min_bounding_point;
  const real3& max_bounding_point = data_manager->measures.collision.max_bounding_point;
  tile_size = data_manager->settings.fluid.kernel_radius * 2;

  real3 diagonal = (fabs(max_bounding_point - min_bounding_point));
  tiles_per_axis = I3(diagonal / R3(tile_size));
  inv_tile_size = 1.0 / tile_size;
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
  ComputeOneLevelGrid();
  OneLevelBroadphase();
}
}
}
