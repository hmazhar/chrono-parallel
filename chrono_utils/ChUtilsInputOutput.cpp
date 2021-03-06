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
// Authors: Radu Serban
// =============================================================================
//
// =============================================================================

#include "assets/ChColorAsset.h"

#include "chrono_utils/ChUtilsInputOutput.h"

namespace chrono {
using namespace collision;
namespace utils {

// -----------------------------------------------------------------------------
// WriteBodies
//
// Write to a CSV file pody position, orientation, and (optionally) linear and
// angular velocity. Optionally, only active bodies are processed.
// -----------------------------------------------------------------------------
void WriteBodies(ChSystem* system,
                 const std::string& filename,
                 bool active_only,
                 bool dump_vel,
                 const std::string& delim) {
  CSV_writer csv(delim);

  for (int i = 0; i < system->Get_bodylist()->size(); i++) {
    ChBody* body = system->Get_bodylist()->at(i);
    if (active_only && !body->IsActive())
      continue;
    csv << body->GetPos() << body->GetRot();
    if (dump_vel)
      csv << body->GetPos_dt() << body->GetWvel_loc();
    csv << std::endl;
  }

  csv.write_to_file(filename);
}

// -----------------------------------------------------------------------------
// WriteCheckpoint
//
// Create a CSV file with a checkpoint ...
//
// -----------------------------------------------------------------------------
bool WriteCheckpoint(ChSystem* system, const std::string& filename) {
  // Infer collision system type (true: parallel, false: bullet)
  bool cd_par = dynamic_cast<collision::ChCollisionSystemParallel*>(system->GetCollisionSystem());

  // Create the CSV stream.
  CSV_writer csv(" ");

  std::vector<ChBody*>::iterator ibody = system->Get_bodylist()->begin();
  for (; ibody != system->Get_bodylist()->end(); ++ibody) {
    ChBody* body = *ibody;

    // Infer body type (0: DVI, 1:DEM)
    int btype = (body->GetContactMethod() == ChBody::DVI) ? 0 : 1;

    // Write body type, body identifier, the body fixed flag, and the collide flag
    csv << btype << body->GetIdentifier() << body->GetBodyFixed() << body->GetCollide();

    // Write collision family information.
    short family_group = 1;
    short family_mask = 0x7FFF;
    if (cd_par) {
      collision::ChCollisionModelParallel* cmodel =
          static_cast<collision::ChCollisionModelParallel*>(body->GetCollisionModel());
      family_group = cmodel->GetFamilyGroup();
      family_mask = cmodel->GetFamilyMask();
    } else {
      collision::ChModelBullet* cmodel = static_cast<collision::ChModelBullet*>(body->GetCollisionModel());
      family_group = cmodel->GetFamilyGroup();
      family_mask = cmodel->GetFamilyMask();
    }
    csv << family_group << family_mask;

    // Write body mass and inertia
    csv << body->GetMass() << body->GetInertiaXX();

    // Write body position, orientation, and their time derivatives
    csv << body->GetPos() << body->GetRot();
    csv << body->GetPos_dt() << body->GetRot_dt();

    csv << std::endl;

    // Write material information
    if (btype == 0) {
      // Write DVI material surface information
      ChSharedPtr<ChMaterialSurface> mat = body->GetMaterialSurface();
      csv << mat->static_friction << mat->sliding_friction << mat->rolling_friction << mat->spinning_friction;
      csv << mat->restitution << mat->cohesion << mat->dampingf;
      csv << mat->compliance << mat->complianceT << mat->complianceRoll << mat->complianceSpin;
    } else {
      // Write DEM material surface information
      ChSharedPtr<ChMaterialSurfaceDEM> mat = body->GetMaterialSurfaceDEM();
      csv << mat->young_modulus << mat->poisson_ratio;
      csv << mat->static_friction << mat->sliding_friction;
      csv << mat->restitution << mat->cohesion;
    }

    csv << std::endl;

    // Count and write all visual assets.
    int num_visual_assets = 0;
    std::vector<ChSharedPtr<ChAsset> >::iterator iasset = (*ibody)->GetAssets().begin();
    for (; iasset != (*ibody)->GetAssets().end(); ++iasset) {
      if ((*iasset).IsType<ChVisualization>())
        num_visual_assets++;
    }
    csv << num_visual_assets << std::endl;

    // Loop over each asset and, for selected visual assets only, write its data
    // on a separate line. If we encounter an unsupported type, return false.
    iasset = (*ibody)->GetAssets().begin();
    for (; iasset != (*ibody)->GetAssets().end(); ++iasset) {
      ChSharedPtr<ChVisualization> visual_asset = (*iasset).DynamicCastTo<ChVisualization>();
      if (visual_asset.IsNull())
        continue;

      // Write relative position and rotation
      csv << visual_asset->Pos << visual_asset->Rot.Get_A_quaternion();

      // Write shape type and geometry data
      if (ChSharedPtr<ChSphereShape> sphere = visual_asset.DynamicCastTo<ChSphereShape>()) {
        csv << collision::SPHERE << sphere->GetSphereGeometry().rad;
      } else if (ChSharedPtr<ChEllipsoidShape> ellipsoid = visual_asset.DynamicCastTo<ChEllipsoidShape>()) {
        csv << collision::ELLIPSOID << ellipsoid->GetEllipsoidGeometry().rad;
      } else if (ChSharedPtr<ChBoxShape> box = visual_asset.DynamicCastTo<ChBoxShape>()) {
        csv << collision::BOX << box->GetBoxGeometry().Size;
      } else if (ChSharedPtr<ChCapsuleShape> capsule = visual_asset.DynamicCastTo<ChCapsuleShape>()) {
        const geometry::ChCapsule& geom = capsule->GetCapsuleGeometry();
        csv << collision::CAPSULE << geom.rad << geom.hlen;
      } else if (ChSharedPtr<ChCylinderShape> cylinder = visual_asset.DynamicCastTo<ChCylinderShape>()) {
        const geometry::ChCylinder& geom = cylinder->GetCylinderGeometry();
        csv << collision::CYLINDER << geom.rad << (geom.p1.y - geom.p2.y) / 2;
      } else if (ChSharedPtr<ChConeShape> cone = visual_asset.DynamicCastTo<ChConeShape>()) {
        const geometry::ChCone& geom = cone->GetConeGeometry();
        csv << collision::CONE << geom.rad.x << geom.rad.y;
      } else if (ChSharedPtr<ChRoundedBoxShape> rbox = visual_asset.DynamicCastTo<ChRoundedBoxShape>()) {
        const geometry::ChRoundedBox& geom = rbox->GetRoundedBoxGeometry();
        csv << collision::ROUNDEDBOX << geom.Size << geom.radsphere;
      } else if (ChSharedPtr<ChRoundedCylinderShape> rcyl = visual_asset.DynamicCastTo<ChRoundedCylinderShape>()) {
        const geometry::ChRoundedCylinder& geom = rcyl->GetRoundedCylinderGeometry();
        csv << collision::ROUNDEDCYL << geom.rad << geom.hlen << geom.radsphere;
      } else {
        // Unsupported visual asset type.
        return false;
      }
      csv << std::endl;
    }
  }

  csv.write_to_file(filename);

  return true;
}

// -----------------------------------------------------------------------------
// ReadCheckpoint
//
// Read a CSV file with checkpoint data and create the bodies.
//
// -----------------------------------------------------------------------------
void ReadCheckpoint(ChSystem* system, const std::string& filename) {
  // Infer system type (true: parallel, false: sequential)
  bool sys_par = dynamic_cast<ChSystemParallelDVI*>(system) || dynamic_cast<ChSystemParallelDEM*>(system);

  // Infer collision system type (true: parallel, false: bullet)
  bool cd_par = dynamic_cast<collision::ChCollisionSystemParallel*>(system->GetCollisionSystem());

  // Open input file stream
  std::ifstream ifile(filename.c_str());
  std::string line;

  while (std::getline(ifile, line)) {
    std::istringstream iss1(line);

    // Read body type, Id, flags
    int btype, bid, bfixed, bcollide;
    short family_group, family_mask;
    iss1 >> btype >> bid >> bfixed >> bcollide >> family_group >> family_mask;

    // Read body mass and inertia
    double mass;
    ChVector<> inertiaXX;
    iss1 >> mass >> inertiaXX.x >> inertiaXX.y >> inertiaXX.z;

    // Read body position, orientation, and their time derivatives
    ChVector<> bpos, bpos_dt;
    ChQuaternion<> brot, brot_dt;
    iss1 >> bpos.x >> bpos.y >> bpos.z >> brot.e0 >> brot.e1 >> brot.e2 >> brot.e3;
    iss1 >> bpos_dt.x >> bpos_dt.y >> bpos_dt.z >> brot_dt.e0 >> brot_dt.e1 >> brot_dt.e2 >> brot_dt.e3;

    // Get the next line in the file (material properties)
    std::getline(ifile, line);
    std::istringstream iss2(line);

    // Create a body of the appropriate type, read and apply material properties
    ChBody* body;
    if (btype == 0) {
      body = (sys_par && cd_par) ? new ChBody(new collision::ChCollisionModelParallel, ChBody::DVI)
                                 : new ChBody(ChBody::DVI);
      ChSharedPtr<ChMaterialSurface> mat = body->GetMaterialSurface();
      iss2 >> mat->static_friction >> mat->sliding_friction >> mat->rolling_friction >> mat->spinning_friction;
      iss2 >> mat->restitution >> mat->cohesion >> mat->dampingf;
      iss2 >> mat->compliance >> mat->complianceT >> mat->complianceRoll >> mat->complianceSpin;
    } else {
      body = (sys_par && cd_par) ? new ChBody(new collision::ChCollisionModelParallel, ChBody::DEM)
                                 : new ChBody(ChBody::DEM);
      ChSharedPtr<ChMaterialSurfaceDEM> mat = body->GetMaterialSurfaceDEM();
      iss2 >> mat->young_modulus >> mat->poisson_ratio;
      iss2 >> mat->static_friction >> mat->sliding_friction;
      iss2 >> mat->restitution >> mat->cohesion;
    }

    // Add the body to the system.
    system->AddBody(ChSharedPtr<ChBody>(body));

    // Set body properties and state
    body->SetPos(bpos);
    body->SetRot(brot);
    body->SetPos_dt(bpos_dt);
    body->SetRot_dt(brot_dt);

    body->SetIdentifier(bid);
    body->SetBodyFixed(bfixed != 0);
    body->SetCollide(bcollide != 0);

    body->SetMass(mass);
    body->SetInertiaXX(inertiaXX);

    // Get next line in the file (number of visualization assets)
    std::getline(ifile, line);
    std::istringstream iss3(line);

    int numAssets;
    iss3 >> numAssets;

    // In a loop, read information about each asset and add geometry to the body
    body->GetCollisionModel()->ClearModel();

    for (int j = 0; j < numAssets; j++) {
      std::getline(ifile, line);
      std::istringstream iss(line);

      // Get relative position and rotation
      ChVector<> apos;
      ChQuaternion<> arot;
      iss >> apos.x >> apos.y >> apos.z >> arot.e0 >> arot.e1 >> arot.e2 >> arot.e3;

      // Get visualization asset type and geometry data.
      // Create the appropriate shape (both visualization and contact).
      int atype;
      iss >> atype;

      switch (collision::ShapeType(atype)) {
        case collision::SPHERE: {
          double radius;
          iss >> radius;
          AddSphereGeometry(body, radius, apos, arot);
        } break;
        case collision::ELLIPSOID: {
          ChVector<> size;
          iss >> size.x >> size.y >> size.z;
          AddEllipsoidGeometry(body, size, apos, arot);
        } break;
        case collision::BOX: {
          ChVector<> size;
          iss >> size.x >> size.y >> size.z;
          AddBoxGeometry(body, size, apos, arot);
        } break;
        case collision::CAPSULE: {
          double radius, hlen;
          iss >> radius >> hlen;
          AddCapsuleGeometry(body, radius, hlen, apos, arot);
        } break;
        case collision::CYLINDER: {
          double radius, hlen;
          iss >> radius >> hlen;
          AddCylinderGeometry(body, radius, hlen, apos, arot);
        } break;
        case collision::CONE: {
          double radius, height;
          iss >> radius >> height;
          AddConeGeometry(body, radius, height, apos, arot);
        } break;
        case collision::ROUNDEDBOX: {
          ChVector<> size;
          double srad;
          iss >> size.x >> size.y >> size.z >> srad;
          AddRoundedBoxGeometry(body, size, srad, apos, arot);
        } break;
        case collision::ROUNDEDCYL: {
          double radius, hlen, srad;
          iss >> radius >> hlen >> srad;
          AddRoundedCylinderGeometry(body, radius, hlen, srad, apos, arot);
        } break;
      }
    }

    // Set the collision family group and the collision family mask.
    if (cd_par) {
      collision::ChCollisionModelParallel* cmodel =
          static_cast<collision::ChCollisionModelParallel*>(body->GetCollisionModel());
      cmodel->SetFamilyGroup(family_group);
      cmodel->SetFamilyMask(family_mask);
    } else {
      collision::ChModelBullet* cmodel = static_cast<collision::ChModelBullet*>(body->GetCollisionModel());
      cmodel->SetFamilyGroup(family_group);
      cmodel->SetFamilyMask(family_mask);
    }

    // Complete construction of the collision model.
    body->GetCollisionModel()->BuildModel();
  }
}

// -----------------------------------------------------------------------------
// WriteShapesPovray
//
// Write CSV output file for PovRay.
// First line contains the number of visual assets and links to follow.
// A line with information about a visualization asset contains:
//    bodyId, bodyActive, x, y, z, e0, e1, e2, e3, shapeType, [shape Data]
// A line with information about a link contains:
//    linkType, [linkData]
//
// NOTE: we do not account for any transform specified for the ChGeometry of
// a visual asset (except for cylinders where that is implicit)!
// -----------------------------------------------------------------------------
void WriteShapesPovray(ChSystem* system, const std::string& filename, bool body_info, const std::string& delim) {
  CSV_writer csv(delim);

  // If requested, Loop over all bodies and write out their position and
  // orientation.  Otherwise, body count is left at 0.
  int b_count = 0;

  if (body_info) {
    std::vector<ChBody*>::iterator ibody = system->Get_bodylist()->begin();
    for (; ibody != system->Get_bodylist()->end(); ++ibody) {
      const ChVector<>& body_pos = (*ibody)->GetFrame_REF_to_abs().GetPos();
      const ChQuaternion<>& body_rot = (*ibody)->GetFrame_REF_to_abs().GetRot();

      csv << (*ibody)->GetIdentifier() << (*ibody)->IsActive() << body_pos << body_rot << std::endl;

      b_count++;
    }
  }

  // Loop over all bodies and over all their assets.
  int a_count = 0;
  std::vector<ChBody*>::iterator ibody = system->Get_bodylist()->begin();
  for (; ibody != system->Get_bodylist()->end(); ++ibody) {
    const ChVector<>& body_pos = (*ibody)->GetFrame_REF_to_abs().GetPos();
    const ChQuaternion<>& body_rot = (*ibody)->GetFrame_REF_to_abs().GetRot();

    ChColor color(0.8f, 0.8f, 0.8f);

    // First loop over assets -- search for a color asset
    std::vector<ChSharedPtr<ChAsset> >::iterator iasset = (*ibody)->GetAssets().begin();
    for (; iasset != (*ibody)->GetAssets().end(); ++iasset) {
      if (ChSharedPtr<ChColorAsset> color_asset = (*iasset).DynamicCastTo<ChColorAsset>())
        color = color_asset->GetColor();
    }

    // Loop over assets once again -- write information for supported types.
    iasset = (*ibody)->GetAssets().begin();
    for (; iasset != (*ibody)->GetAssets().end(); ++iasset) {
      ChSharedPtr<ChVisualization> visual_asset = (*iasset).DynamicCastTo<ChVisualization>();
      if (visual_asset.IsNull())
        continue;

      const Vector& asset_pos = visual_asset->Pos;
      Quaternion asset_rot = visual_asset->Rot.Get_A_quaternion();

      Vector pos = body_pos + body_rot.Rotate(asset_pos);
      Quaternion rot = body_rot % asset_rot;

      std::stringstream gss;

      if (ChSharedPtr<ChSphereShape> sphere = visual_asset.DynamicCastTo<ChSphereShape>()) {
        gss << collision::SPHERE << delim << sphere->GetSphereGeometry().rad;
        a_count++;
      } else if (ChSharedPtr<ChEllipsoidShape> ellipsoid = visual_asset.DynamicCastTo<ChEllipsoidShape>()) {
        const Vector& size = ellipsoid->GetEllipsoidGeometry().rad;
        gss << collision::ELLIPSOID << delim << size.x << delim << size.y << delim << size.z;
        a_count++;
      } else if (ChSharedPtr<ChBoxShape> box = visual_asset.DynamicCastTo<ChBoxShape>()) {
        const Vector& size = box->GetBoxGeometry().Size;
        gss << collision::BOX << delim << size.x << delim << size.y << delim << size.z;
        a_count++;
      } else if (ChSharedPtr<ChCapsuleShape> capsule = visual_asset.DynamicCastTo<ChCapsuleShape>()) {
        const geometry::ChCapsule& geom = capsule->GetCapsuleGeometry();
        gss << collision::CAPSULE << delim << geom.rad << delim << geom.hlen;
        a_count++;
      } else if (ChSharedPtr<ChCylinderShape> cylinder = visual_asset.DynamicCastTo<ChCylinderShape>()) {
        const geometry::ChCylinder& geom = cylinder->GetCylinderGeometry();
        gss << collision::CYLINDER << delim << geom.rad << delim << geom.p1.x << delim << geom.p1.y << delim
            << geom.p1.z << delim << geom.p2.x << delim << geom.p2.y << delim << geom.p2.z;
        a_count++;
      } else if (ChSharedPtr<ChConeShape> cone = visual_asset.DynamicCastTo<ChConeShape>()) {
        const geometry::ChCone& geom = cone->GetConeGeometry();
        gss << collision::CONE << delim << geom.rad.x << delim << geom.rad.y;
        a_count++;
      } else if (ChSharedPtr<ChRoundedBoxShape> rbox = visual_asset.DynamicCastTo<ChRoundedBoxShape>()) {
        const geometry::ChRoundedBox& geom = rbox->GetRoundedBoxGeometry();
        gss << collision::ROUNDEDBOX << delim << geom.Size.x << delim << geom.Size.y << delim << geom.Size.z << delim
            << geom.radsphere;
        a_count++;
      } else if (ChSharedPtr<ChRoundedCylinderShape> rcyl = visual_asset.DynamicCastTo<ChRoundedCylinderShape>()) {
        const geometry::ChRoundedCylinder& geom = rcyl->GetRoundedCylinderGeometry();
        gss << collision::ROUNDEDCYL << delim << geom.rad << delim << geom.hlen << delim << geom.radsphere;
        a_count++;
      } else if (ChSharedPtr<ChTriangleMeshShape> mesh = visual_asset.DynamicCastTo<ChTriangleMeshShape>()) {
        gss << collision::TRIANGLEMESH << delim << "\"" << mesh->GetName() << "\"";
        a_count++;
      }

      csv << (*ibody)->GetIdentifier() << (*ibody)->IsActive() << pos << rot << color << gss.str() << std::endl;
    }
  }

  // Loop over all links.  Write information on selected types of links.
  int l_count = 0;
  std::vector<ChLink*>::iterator ilink = system->Get_linklist()->begin();
  for (; ilink != system->Get_linklist()->end(); ++ilink) {
    int type = (*ilink)->GetType();

    if (ChLinkLockRevolute* link = dynamic_cast<ChLinkLockRevolute*>(*ilink)) {
      chrono::ChFrame<> frA_abs = *(link->GetMarker1()) >> *(link->GetBody1());
      chrono::ChFrame<> frB_abs = *(link->GetMarker2()) >> *(link->GetBody2());

      csv << type << frA_abs.GetPos() << frA_abs.GetA().Get_A_Zaxis() << std::endl;
      l_count++;
    } else if (ChLinkLockSpherical* link = dynamic_cast<ChLinkLockSpherical*>(*ilink)) {
      chrono::ChFrame<> frA_abs = *(link->GetMarker1()) >> *(link->GetBody1());
      chrono::ChFrame<> frB_abs = *(link->GetMarker2()) >> *(link->GetBody2());

      csv << type << frA_abs.GetPos() << std::endl;
      l_count++;
    }
    if (ChLinkLockPrismatic* link = dynamic_cast<ChLinkLockPrismatic*>(*ilink)) {
      chrono::ChFrame<> frA_abs = *(link->GetMarker1()) >> *(link->GetBody1());
      chrono::ChFrame<> frB_abs = *(link->GetMarker2()) >> *(link->GetBody2());

      csv << type << frA_abs.GetPos() << frA_abs.GetA().Get_A_Zaxis() << std::endl;
      l_count++;
    } else if (ChLinkUniversal* link = dynamic_cast<ChLinkUniversal*>(*ilink)) {
      chrono::ChFrame<> frA_abs = link->GetFrame1Abs();
      chrono::ChFrame<> frB_abs = link->GetFrame2Abs();

      csv << type << frA_abs.GetPos() << frA_abs.GetA().Get_A_Xaxis() << frB_abs.GetA().Get_A_Yaxis() << std::endl;
      l_count++;
    } else if (ChLinkSpring* link = dynamic_cast<ChLinkSpring*>(*ilink)) {
      chrono::ChFrame<> frA_abs = *(link->GetMarker1()) >> *(link->GetBody1());
      chrono::ChFrame<> frB_abs = *(link->GetMarker2()) >> *(link->GetBody2());

      csv << type << frA_abs.GetPos() << frB_abs.GetPos() << std::endl;
      l_count++;
    } else if (ChLinkSpringCB* link = dynamic_cast<ChLinkSpringCB*>(*ilink)) {
      chrono::ChFrame<> frA_abs = *(link->GetMarker1()) >> *(link->GetBody1());
      chrono::ChFrame<> frB_abs = *(link->GetMarker2()) >> *(link->GetBody2());

      csv << type << frA_abs.GetPos() << frB_abs.GetPos() << std::endl;
      l_count++;
    } else if (ChLinkDistance* link = dynamic_cast<ChLinkDistance*>(*ilink)) {
      csv << type << link->GetEndPoint1Abs() << link->GetEndPoint2Abs() << std::endl;
      l_count++;
    } else if (ChLinkEngine* link = dynamic_cast<ChLinkEngine*>(*ilink)) {
      chrono::ChFrame<> frA_abs = *(link->GetMarker1()) >> *(link->GetBody1());
      chrono::ChFrame<> frB_abs = *(link->GetMarker2()) >> *(link->GetBody2());

      csv << type << frA_abs.GetPos() << frA_abs.GetA().Get_A_Zaxis() << std::endl;
      l_count++;
    }
  }

  // Write the output file, including a first line with number of bodies, visual
  // assets, and links.
  std::stringstream header;
  header << b_count << delim << a_count << delim << l_count << delim << std::endl;

  csv.write_to_file(filename, header.str());
}

// -----------------------------------------------------------------------------
// WriteMeshPovray
//
// Write the triangular mesh from the specified OBJ file as a macro in a PovRay
// include file.
// -----------------------------------------------------------------------------
void WriteMeshPovray(const std::string& obj_filename,
                     const std::string& mesh_name,
                     const std::string& out_dir,
                     const ChColor& col,
                     const ChVector<>& pos,
                     const ChQuaternion<>& rot) {
  // Read trimesh from OBJ file
  geometry::ChTriangleMeshConnected trimesh;
  trimesh.LoadWavefrontMesh(obj_filename, false, false);

  // Transform vertices.
  for (int i = 0; i < trimesh.m_vertices.size(); i++)
    trimesh.m_vertices[i] = pos + rot.Rotate(trimesh.m_vertices[i]);

  // Open output file.
  std::string pov_filename = out_dir + "/" + mesh_name + ".inc";
  std::ofstream ofile(pov_filename.c_str());

  ofile << "#declare " << mesh_name << "_mesh = mesh2 {" << std::endl;

  // Write vertices.
  ofile << "vertex_vectors {" << std::endl;
  ofile << trimesh.m_vertices.size();
  for (int i = 0; i < trimesh.m_vertices.size(); i++) {
    ChVector<> v = trimesh.m_vertices[i];
    ofile << ",\n<" << v.x << ", " << v.z << ", " << v.y << ">";
  }
  ofile << "\n}" << std::endl;

  // Write face connectivity.
  ofile << "face_indices {" << std::endl;
  ofile << trimesh.m_face_v_indices.size();
  for (int i = 0; i < trimesh.m_face_v_indices.size(); i++) {
    ChVector<int> face = trimesh.m_face_v_indices[i];
    ofile << ",\n<" << face.x << ", " << face.y << ", " << face.z << ">";
  }
  ofile << "\n}" << std::endl;

  ofile << "\n}" << std::endl;

  // Write the object
  ofile << "#declare " << mesh_name << " = object {" << std::endl;

  ofile << "   " << mesh_name << "_mesh" << std::endl;
  ofile << "   texture {" << std::endl;
  ofile << "      pigment {color rgb<" << col.R << ", " << col.G << ", " << col.B << ">}" << std::endl;
  ofile << "      finish  {phong 0.2  diffuse 0.6}" << std::endl;
  ofile << "    }" << std::endl;
  ofile << "}" << std::endl;
}

void WriteConvexShapes(const std::string& obj_filename, ChConvexDecompositionHACDv2& convex_shape) {
  ChConvexDecomposition* used_decomposition = &convex_shape;

  int hull_count = used_decomposition->GetHullCount();
  std::ofstream ofile(obj_filename.c_str());
  ofile << hull_count << " ";
  for (int c = 0; c < hull_count; c++) {
    std::vector<ChVector<double> > convexhull;
    used_decomposition->GetConvexHullResult(c, convexhull);
    ofile << convexhull.size() << " ";
    for (int i = 0; i < convexhull.size(); i++) {
      ofile << convexhull[i].x << " " << convexhull[i].y << " " << convexhull[i].z << " ";
    }
  }
  ofile.close();
}

void ReadConvexShapes(const std::string& obj_filename, std::vector<std::vector<ChVector<double> > >& convex_hulls) {
  std::ifstream ifile(obj_filename.c_str());

  int hull_count;
  ifile >> hull_count;
  convex_hulls.resize(hull_count);

  for (int c = 0; c < hull_count; c++) {
    int hull_size;
    ifile >> hull_size;
    std::vector<ChVector<double> > convexhull(hull_size);

    for (int i = 0; i < hull_size; i++) {
      ifile >> convexhull[i].x >> convexhull[i].y >> convexhull[i].z;
    }
    convex_hulls[c] = convexhull;
  }
  if(ifile.fail()){
    std::cout<<"FILE FAIL"<<std::endl;
  }
  ifile.close();
}

}  // namespace utils
}  // namespace chrono
