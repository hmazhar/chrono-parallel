#include <algorithm>

// not used but prevents compilation errors with cuda 7 RC
#include <thrust/transform.h>

#include "chrono_parallel/collision/ChCAABBGenerator.h"
using namespace chrono::collision;

__constant__ uint numAABB_const;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
static __device__ __host__ void ComputeAABBSphere(const real& radius, const real3& position, real3& minp, real3& maxp) {
  minp = position - R3(radius);
  maxp = position + R3(radius);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
static __device__ __host__ void ComputeAABBTriangle(const real3& A,
                                                    const real3& B,
                                                    const real3& C,
                                                    real3& minp,
                                                    real3& maxp) {
  minp.x = std::min(A.x, std::min(B.x, C.x));
  minp.y = std::min(A.y, std::min(B.y, C.y));
  minp.z = std::min(A.z, std::min(B.z, C.z));
  maxp.x = std::max(A.x, std::max(B.x, C.x));
  maxp.y = std::max(A.y, std::max(B.y, C.y));
  maxp.z = std::max(A.z, std::max(B.z, C.z));
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
static void ComputeAABBBox(const real3& dim,
                           const real3& lpositon,
                           const real3& positon,
                           const real4& lrotation,
                           const real4& rotation,
                           real3& minp,
                           real3& maxp) {
  real4 q1 = mult(rotation, lrotation);
  M33 rotmat = AMat(q1);
  rotmat = AbsMat(rotmat);

  real3 temp = MatMult(rotmat, dim);

  real3 pos = quatRotate(lpositon, rotation) + positon;
  minp = pos - temp;
  maxp = pos + temp;

  // cout<<minp.x<<" "<<minp.y<<" "<<minp.z<<"  |  "<<maxp.x<<" "<<maxp.y<<" "<<maxp.z<<endl;
  //    real3 pos = quatRotate(lpositon, rotation) + positon; //new position
  //    real4 q1 = mult(rotation, lrotation); //full rotation
  //    real4 q = R4(q1.y, q1.z, q1.w, q1.x);
  //    real t[3] = { pos.x, pos.y, pos.z };
  //    real mina[3] = { -dim.x, -dim.y, -dim.z };
  //    real maxa[3] = { dim.x, dim.y, dim.z };
  //    real minb[3] = { 0, 0, 0 };
  //    real maxb[3] = { 0, 0, 0 };
  //    real m[3][3];
  //    real qx2 = q.x * q.x;
  //    real qy2 = q.y * q.y;
  //    real qz2 = q.z * q.z;
  //    m[0][0] = 1 - 2 * qy2 - 2 * qz2;
  //    m[1][0] = 2 * q.x * q.y + 2 * q.z * q.w;
  //    m[2][0] = 2 * q.x * q.z - 2 * q.y * q.w;
  //    m[0][1] = 2 * q.x * q.y - 2 * q.z * q.w;
  //    m[1][1] = 1 - 2 * qx2 - 2 * qz2;
  //    m[2][1] = 2 * q.y * q.z + 2 * q.x * q.w   ;
  //    m[0][2] = 2 * q.x * q.z + 2 * q.y * q.w;
  //    m[1][2] = 2 * q.y * q.z - 2 * q.x * q.w;
  //    m[2][2] = 1 - 2 * qx2 - 2 * qy2;
  //
  //    // For all three axes
  //    for (int i = 0; i < 3; i++) {
  //        // Start by adding in translation
  //        minb[i] = maxb[i] = t[i];
  //
  //        // Form extent by summing smaller and larger terms respectively
  //        for (int j = 0; j < 3; j++) {
  //            real e = m[i][j] * mina[j];
  //            real f = m[i][j] * maxa[j];
  //
  //            if (e < f) {
  //                minb[i] += e;
  //                maxb[i] += f;
  //            } else {
  //                minb[i] += f;
  //                maxb[i] += e;
  //            }
  //        }
  //    }

  //    minp = R3(minb[0], minb[1], minb[2]);
  //    maxp = R3(maxb[0], maxb[1], maxb[2]);
}

static void ComputeAABBCone(const real3& dim,
                            const real3& lpositon,
                            const real3& positon,
                            const real4& lrotation,
                            const real4& rotation,
                            real3& minp,
                            real3& maxp) {
  real4 q1 = mult(rotation, lrotation);
  M33 rotmat = AMat(q1);
  rotmat = AbsMat(rotmat);

  real3 temp = MatMult(rotmat, R3(dim.x, dim.y, dim.z / 2.0));

  real3 pos = quatRotate(lpositon - R3(0, 0, dim.z / 2.0), rotation) + positon;
  minp = pos - temp;
  maxp = pos + temp;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
static void ComputeAABBConvex(const real3* convex_points,
                              const real3& B,
                              const real3& lpos,
                              const real3& pos,
                              const real4& rot,
                              real3& minp,
                              real3& maxp) {
  int start = B.y;
  int size = B.x;
  real3 point_0 = quatRotate(convex_points[start] + lpos, rot) + pos;

  minp = maxp = point_0;
  for (int i = start; i < start + size; i++) {
    real3 p = quatRotate(convex_points[i] + lpos, rot) + pos;
    if (minp.x > p.x) {
      minp.x = p.x;
    }
    if (minp.y > p.y) {
      minp.y = p.y;
    }
    if (minp.z > p.z) {
      minp.z = p.z;
    }
    if (maxp.x < p.x) {
      maxp.x = p.x;
    }
    if (maxp.y < p.y) {
      maxp.y = p.y;
    }
    if (maxp.z < p.z) {
      maxp.z = p.z;
    }
  }

  minp = minp - R3(B.z);
  maxp = maxp + R3(B.z);
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__device__ __host__ void function_ComputeAABB(const uint& index,
                                              const shape_type* obj_data_T,
                                              const real3* obj_data_A,
                                              const real3* obj_data_B,
                                              const real3* obj_data_C,
                                              const real4* obj_data_R,
                                              const uint* obj_data_ID,
                                              const real3* convex_points,
                                              const real3* body_pos,
                                              const real4* body_rot,
                                              const uint& numAABB,
                                              const real& collision_envelope,
                                              real3* aabb_data) {
  shape_type type = obj_data_T[index];
  uint id = obj_data_ID[index];
  real3 A = obj_data_A[index];
  real3 B = obj_data_B[index];
  real3 C = obj_data_C[index];
  real3 position = body_pos[id];
  real4 rotation = (mult(body_rot[id], obj_data_R[index]));
  real3 temp_min;
  real3 temp_max;

  if (type == SPHERE) {
    A = quatRotate(A, body_rot[id]);
    ComputeAABBSphere(B.x + collision_envelope, A + position, temp_min, temp_max);
    // if(id==6){
    // cout<<temp_min.x<<" "<<temp_min.y<<" "<<temp_min.z<<"  |  "<<temp_max.x<<" "<<temp_max.y<<" "<<temp_max.z<<endl;

    //}
  } else if (type == TRIANGLEMESH) {
    A = quatRotate(A, body_rot[id]) + position;
    B = quatRotate(B, body_rot[id]) + position;
    C = quatRotate(C, body_rot[id]) + position;
    ComputeAABBTriangle(A, B, C, temp_min, temp_max);
  } else if (type == ELLIPSOID || type == BOX || type == CYLINDER || type == CONE) {
    ComputeAABBBox(B + collision_envelope, A, position, obj_data_R[index], body_rot[id], temp_min, temp_max);
  } else if (type == ROUNDEDBOX || type == ROUNDEDCYL || type == ROUNDEDCONE) {
    ComputeAABBBox(B + C.x + collision_envelope, A, position, obj_data_R[index], body_rot[id], temp_min, temp_max);
  } else if (type == CAPSULE) {
    real3 B_ = R3(B.x, B.x + B.y, B.z) + collision_envelope;
    ComputeAABBBox(B_, A, position, obj_data_R[index], body_rot[id], temp_min, temp_max);
  } else if (type == CONVEX) {
    ComputeAABBConvex(convex_points, B, A, position, rotation, temp_min, temp_max);
    temp_min -= collision_envelope;
    temp_max += collision_envelope;
  } else {
    return;
  }

  aabb_data[index] = temp_min;
  aabb_data[index + numAABB] = temp_max;
}

void ChCAABBGenerator::host_ComputeAABB(const shape_type* obj_data_T,
                                        const real3* obj_data_A,
                                        const real3* obj_data_B,
                                        const real3* obj_data_C,
                                        const real4* obj_data_R,
                                        const uint* obj_data_ID,
                                        const real3* convex_data,
                                        const real3* body_pos,
                                        const real4* body_rot,
                                        const real collision_envelope,
                                        real3* aabb_data) {
#pragma omp parallel for

  for (int i = 0; i < numAABB; i++) {
    function_ComputeAABB(i, obj_data_T, obj_data_A, obj_data_B, obj_data_C, obj_data_R, obj_data_ID, convex_data,
                         body_pos, body_rot, numAABB, collision_envelope, aabb_data);
  }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ChCAABBGenerator::ChCAABBGenerator() {
}

void ChCAABBGenerator::GenerateAABB(const custom_vector<shape_type>& obj_data_T,
                                    const custom_vector<real3>& obj_data_A,
                                    const custom_vector<real3>& obj_data_B,
                                    const custom_vector<real3>& obj_data_C,
                                    const custom_vector<real4>& obj_data_R,
                                    const custom_vector<uint>& obj_data_ID,
                                    const custom_vector<real3>& convex_data,
                                    const custom_vector<real3>& body_pos,
                                    const custom_vector<real4>& body_rot,
                                    const real collision_envelope,
                                    custom_vector<real3>& aabb_data) {

  LOG(TRACE) << "AABB START";


  numAABB = obj_data_T.size();
  aabb_data.resize(numAABB * 2);
#ifdef SIM_ENABLE_GPU_MODE
  COPY_TO_CONST_MEM(numAABB);
  device_ComputeAABB __KERNEL__(BLOCKS(numAABB), THREADS)(
      CASTS(obj_data_T.data()), CASTR3(obj_data_A.data()), CASTR3(obj_data_B.data()), CASTR3(obj_data_C.data()),
      CASTR4(obj_data_R.data()), CASTU1(obj_data_ID.data()), CASTR3(body_pos.data()), CASTR4(body_rot.data()),
      CASTR3(aabb_data.data()));
#else
  host_ComputeAABB(obj_data_T.data(), obj_data_A.data(), obj_data_B.data(), obj_data_C.data(), obj_data_R.data(),
                   obj_data_ID.data(), convex_data.data(), body_pos.data(), body_rot.data(), collision_envelope,
                   aabb_data.data());
#endif

  LOG(TRACE) << "AABB END";

#if PRINT_LEVEL == 2
//    for(int i=0; i<numAABB; i++){
//
//    	cout<<real3(aabb_data[i]).x<<" "<<real3(aabb_data[i]).y<<" "<<real3(aabb_data[i]).z<<endl;
//    	cout<<real3(aabb_data[i+numAABB]).x<<" "<<real3(aabb_data[i+numAABB]).y<<"
//    "<<real3(aabb_data[i+numAABB]).z<<endl;
//    	cout<<"-------"<<endl;
//    }

#endif
}
