#ifndef CHC_MODELGPU_H
#define CHC_MODELGPU_H

//////////////////////////////////////////////////
//
//   ChCCollisionModelGPU.h
//
//   A wrapper to use the GPU collision detection
//
//   HEADER file for CHRONO,
//   Multibody dynamics engine
//
// ------------------------------------------------
//   Copyright:Alessandro Tasora / DeltaKnowledge
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "collision/ChCCollisionModel.h"
#include "ChApiParallel.h"
#include "ChParallelMath.h"
#include "ChParallelDefines.h"
using namespace std;
namespace chrono {
// forward references
class ChBody;

namespace collision {
///  A wrapper to uses GPU collision detection

class ChApiGPU ChCollisionModelParallel: public ChCollisionModel {
	public:

		ChCollisionModelParallel();
		virtual ~ChCollisionModelParallel();

		/// Deletes all inserted geometries.
		/// Also, if you begin the definition of a model, AFTER adding
		/// the geometric description, remember to call the ClearModel().
		/// MUST be inherited by child classes! (ex for resetting also BV hierarchies)
		virtual int ClearModel();

		/// Builds the BV hierarchy.
		/// Call this function AFTER adding the geometric description.
		/// MUST be inherited by child classes! (ex for bulding BV hierarchies)
		virtual int BuildModel();

		/// Sets the position and orientation of the collision
		/// model as the rigid body current position.
		virtual void SyncPosition();

		/// Gets the pointer to the client owner ChPhysicsItem.
		virtual ChPhysicsItem *GetPhysicsItem();
		//
		// GEOMETRY DESCRIPTION
		//
		//  The following functions must be called inbetween
		//  the ClearModel() BuildModel() pair.

		/// Add a sphere shape to this model, for collision purposes
		virtual bool AddSphere(double radius, ChVector<> *pos = 0);

		/// Add an ellipsoid shape to this model, for collision purposes
		virtual bool AddEllipsoid(double rx, double ry, double rz, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add a box shape to this model, for collision purposes
		virtual bool AddBox(double hx, double hy, double hz, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add a triangle shape to this model, for collision purposes
		virtual bool AddTriangle(ChVector<> A, ChVector<> B, ChVector<> C, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add a cylinder to this model (default axis on Y direction), for collision purposes
		virtual bool AddCylinder(double rx, double ry, double rz, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add a cone to this model (default axis on Y direction), for collision purposes
		virtual bool AddCone(double rx, double ry, double rz, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add a 2d rectangle shape to this model, for collision purposes
		virtual bool AddRectangle(double rx, double ry);
		/// Add a 2d disc to this model, for collision purposes
		virtual bool AddDisc(double rad);
		/// Add a 2d ellipse to this model, for collision purposes
		virtual bool AddEllipse(double rx, double ry);
		/// Add a capsule to this model, for collision purposes
		virtual bool AddCapsule(double len, double rad);
		/// Add a cone to this model, for collision purposes
		virtual bool AddCone(double rad, double h);

		virtual bool AddConvexHull(std::vector<ChVector<double> > &pointlist, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add a triangle mesh to this model, passing a triangle mesh (do not delete the triangle mesh
		/// until the collision model, because depending on the implementation of inherited ChCollisionModel
		/// classes, maybe the triangle is referenced via a striding interface or just copied)
		/// Note: if possible, in sake of high performance, avoid triangle meshes and prefer simplified
		/// representations as compounds of convex shapes of boxes/spheres/etc.. type.
		virtual bool AddTriangleMesh(const geometry::ChTriangleMesh &trimesh,  ///< the triangle mesh
				bool is_static,         ///< true only if model doesn't move (es.a terrain). May improve performance
				bool is_convex,         ///< true if mesh convex hull is used (only for simple mesh). May improve robustness
				ChVector<> *pos = 0,
				ChMatrix33<> *rot = 0 ///< displacement respect to COG (optional)
				);

		/// Add a barrel-like shape to this model (main axis on Y direction), for collision purposes.
		/// The barrel shape is made by lathing an arc of an ellipse around the vertical Y axis.
		/// The center of the ellipse is on Y=0 level, and it is ofsetted by R_offset from
		/// the Y axis in radial direction. The two radii of the ellipse are R_vert (for the
		/// vertical direction, i.e. the axis parellel to Y) and R_hor (for the axis that
		/// is perpendicular to Y). Also, the solid is clamped with two discs on the top and
		/// the bottom, at levels Y_low and Y_high.
		virtual bool AddBarrel(double Y_low, double Y_high, double R_vert, double R_hor, double R_offset, ChVector<> *pos = 0, ChMatrix33<> *rot = 0);

		/// Add all shapes already contained in another model.
		/// Thank to the adoption of shared pointers, underlying shapes are
		/// shared (not copied) among the models; this will save memory when you must
		/// simulate thousands of objects with the same collision shape.
		/// The 'another' model must be of ChModelBullet subclass.
		virtual bool AddCopyOfAnotherModel(ChCollisionModel *another);

		///returns the axis aligned bounding box for this collision model
		virtual void GetAABB(ChVector<> &bbmin, ChVector<> &bbmax) const;
		/// Set the collision family for this object
		virtual void SetFamily(int mfamily);
		/// Get the collision family for this object
		virtual int GetFamily();

		/// Set the collision family mask for families to not collide with
		virtual void SetFamilyMaskNoCollisionWithFamily(int mfamily);
		/// Set the collision family mask for families to collide with
		virtual void SetFamilyMaskDoCollisionWithFamily(int mfamily);
		/// Check if the model collides with a family
		virtual bool GetFamilyMaskDoesCollisionWithFamily(int mfamily);
		/// Return a pointer to the body
		ChBody *GetBody() const {
			return mbody;
		}
		;

		/// Get the number of objects in this model
		int GetNObjects() {
			return nObjects;
		}
		///Get family to not collide with
		int GetNoCollFamily();

		struct bData {
				real3 A;
				real3 B;
				real3 C;
				real4 R;
				int type;
		};

		vector<bData> mData;

	protected:

		real3 inertia;
		unsigned int nObjects;
		int colFam, noCollWith;
};

} // END_OF_NAMESPACE____
} // END_OF_NAMESPACE____
#endif
