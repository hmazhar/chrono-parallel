#ifndef CH_UTILS_CREATORS_H
#define CH_UTILS_CREATORS_H

#include <cmath>
#include <vector>
#include <string>

#include "core/ChSmartpointers.h"
#include "core/ChVector.h"
#include "core/ChQuaternion.h"

#include "physics/ChSystem.h"
#include "physics/ChSystemDEM.h"
#include "physics/ChBody.h"
#include "physics/ChBodyDEM.h"
#include "physics/ChMaterialSurface.h"
#include "physics/ChMaterialSurfaceDEM.h"

#include "assets/ChSphereShape.h"
#include "assets/ChEllipsoidShape.h"
#include "assets/ChBoxShape.h"
#include "assets/ChCapsuleShape.h"
#include "assets/ChCylinderShape.h"
#include "assets/ChConeShape.h"
#include "assets/ChTriangleMeshShape.h"

#include "chrono_utils/ChApiUtils.h"
#include "chrono_utils/ChUtilsCommon.h"


namespace chrono {
namespace utils {


// -------------------------------------------------------------------------------
// AddSphereGeometry
// AddEllipsoidGeometry
// AddBoxGeometry
// AddCapsuleGeometry
// AddCylinderGeometry
// AddConeGeometry
// AddTriangleMeshGeometry
//
// Utility functions for adding contact and asset geometry shapes to a body
// -------------------------------------------------------------------------------
inline
void AddSphereGeometry(ChBody*               body,
                       double                radius,
                       const ChVector<>&     pos = ChVector<>(0,0,0),
                       const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	body->GetCollisionModel()->AddSphere(radius, pos);

	ChSharedPtr<ChSphereShape> sphere(new ChSphereShape);
	sphere->GetSphereGeometry().rad = radius;
	sphere->Pos = pos;
	sphere->Rot = rot;

	body->GetAssets().push_back(sphere);
}

inline
void AddEllipsoidGeometry(ChBody*               body,
                          const ChVector<>&     size,
                          const ChVector<>&     pos = ChVector<>(0,0,0),
                          const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	body->GetCollisionModel()->AddEllipsoid(size.x, size.y, size.z, pos, rot);

	ChSharedPtr<ChEllipsoidShape> ellipsoid(new ChEllipsoidShape);
	ellipsoid->GetEllipsoidGeometry().rad = size;
	ellipsoid->Pos = pos;
	ellipsoid->Rot = rot;

	body->GetAssets().push_back(ellipsoid);
}

inline
void AddBoxGeometry(ChBody*               body,
                    const ChVector<>&     size,
                    const ChVector<>&     pos = ChVector<>(0,0,0),
                    const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	body->GetCollisionModel()->AddBox(size.x, size.y, size.z, pos, rot);

	ChSharedPtr<ChBoxShape> box(new ChBoxShape);
	box->GetBoxGeometry().Size = size;
	box->Pos = pos;
	box->Rot = rot;

	body->GetAssets().push_back(box);
}

inline
void AddCapsuleGeometry(ChBody*               body,
                        double                radius,
                        double                hlen,
                        const ChVector<>&     pos = ChVector<>(0,0,0),
                        const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	body->GetCollisionModel()->AddCapsule(radius, hlen, pos, rot);

	ChSharedPtr<ChCapsuleShape> capsule(new ChCapsuleShape);
	capsule->GetCapsuleGeometry().rad = radius;
	capsule->GetCapsuleGeometry().hlen = hlen;
	capsule->Pos = pos;
	capsule->Rot = rot;

	body->GetAssets().push_back(capsule);
}

inline
void AddCylinderGeometry(ChBody*               body,
                         double                radius,
                         double                height,
                         const ChVector<>&     pos = ChVector<>(0,0,0),
                         const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	body->GetCollisionModel()->AddCylinder(radius, radius, height, pos, rot);

	ChSharedPtr<ChCylinderShape> cylinder(new ChCylinderShape);
	cylinder->GetCylinderGeometry().rad = radius;
	cylinder->GetCylinderGeometry().p1 = ChVector<>(0,  height / 2, 0);
	cylinder->GetCylinderGeometry().p2 = ChVector<>(0, -height / 2, 0);
	cylinder->Pos = pos;
	cylinder->Rot = rot;

	body->GetAssets().push_back(cylinder);
}

inline
void AddConeGeometry(ChBody*               body,
                     double                radius,
                     double                height,
                     const ChVector<>&     pos = ChVector<>(0,0,0),
                     const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	body->GetCollisionModel()->AddCone(radius, radius, height, pos, rot);

	ChSharedPtr<ChConeShape> cone(new ChConeShape);
	cone->GetConeGeometry().rad = ChVector<>(radius, height, radius);
	cone->Pos = pos;
	cone->Rot = rot;

	body->GetAssets().push_back(cone);
}

inline
void AddTriangleMeshGeometry(ChBody*               body,
                             const std::string&    obj_filename,
                             const std::string&    name,
                             const ChVector<>&     pos = ChVector<>(0,0,0),
                             const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
	ChTriangleMeshConnected trimesh;
	trimesh.LoadWavefrontMesh(obj_filename, false, false);

	for (int i = 0; i < trimesh.m_vertices.size(); i++)
		trimesh.m_vertices[i] = pos + rot.Rotate(trimesh.m_vertices[i]);

	body->GetCollisionModel()->AddTriangleMesh(trimesh, false, false);

	ChSharedPtr<ChTriangleMeshShape> trimesh_shape(new ChTriangleMeshShape);
	trimesh_shape->SetMesh(trimesh);
	trimesh_shape->SetName(name);
	trimesh_shape->Pos = ChVector<>(0,0,0);
	trimesh_shape->Rot = ChQuaternion<>(1,0,0,0);

	body->GetAssets().push_back(trimesh_shape);
}

// -------------------------------------------------------------------------------
// CreateBoxContainerDEM
// CreateBoxContainerDVI
//
// Create a fixed body with contact and asset geometry representing a box with 5
// walls (no top).
// -------------------------------------------------------------------------------
CH_UTILS_API
void CreateBoxContainerDEM(ChSystem*                           system,
                           int                                 id,
                           ChSharedPtr<ChMaterialSurfaceDEM>&  mat,
                           const ChVector<>&                   hdim,
                           double                              hthick,
                           const ChVector<>&                   pos = ChVector<>(0,0,0),
                           const ChQuaternion<>&               rot = ChQuaternion<>(1,0,0,0),
                           bool                                collide = true);

CH_UTILS_API
void CreateBoxContainerDVI(ChSystem*                           system,
                           int                                 id,
                           ChSharedPtr<ChMaterialSurface>&     mat,
                           const ChVector<>&                   hdim,
                           double                              hthick,
                           const ChVector<>&                   pos = ChVector<>(0,0,0),
                           const ChQuaternion<>&               rot = ChQuaternion<>(1,0,0,0),
                           bool                                collide = true);


} // end namespace utils
} // end namespace chrono


#endif
