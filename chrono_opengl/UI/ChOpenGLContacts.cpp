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
// Renders contact points as a point cloud
// =============================================================================

#include <iostream>
#include "chrono_opengl/UI/ChOpenGLContacts.h"
#include "chrono_opengl/ChOpenGLMaterials.h"
namespace chrono {
using namespace collision;
namespace opengl {
using namespace glm;

ChOpenGLContacts::ChOpenGLContacts() {
}

bool ChOpenGLContacts::Initialize(ChOpenGLMaterial mat, ChOpenGLShader* shader) {
  if (this->GLReturnedError("Contacts::Initialize - on entry"))
    return false;
  contact_data.push_back(vec3(0, 0, 0));
  contacts.Initialize(contact_data, mat, shader);
  contacts.SetPointSize(0.01);
}

void ChOpenGLContacts::UpdateChrono(ChSystem* system) {
}
void ChOpenGLContacts::UpdateChronoParallel(ChSystemParallel* system) {
  ChParallelDataManager* data_manager = system->data_manager;
  uint num_rigid_contacts = data_manager->num_rigid_contacts;
  uint num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
  uint num_fluid_contacts = data_manager->num_fluid_contacts;
  uint num_total_contacts = num_rigid_contacts + num_rigid_fluid_contacts;
  if (num_total_contacts == 0) {
    return;
  }

  contact_data.resize(num_rigid_contacts * 2 + num_rigid_fluid_contacts);

#pragma omp parallel for
  for (int i = 0; i < num_rigid_contacts; i++) {
    real3 cpta = data_manager->host_data.cpta_rigid_rigid[i];
    real3 cptb = data_manager->host_data.cptb_rigid_rigid[i];

    contact_data[i] = glm::vec3(cpta.x, cpta.y, cpta.z);
    contact_data[i + data_manager->num_rigid_contacts] = glm::vec3(cptb.x, cptb.y, cptb.z);
  }

#pragma omp parallel for
  for (int i = 0; i < num_rigid_fluid_contacts; i++) {
    real3 cpta = data_manager->host_data.cpta_rigid_fluid[i];
    contact_data[i + data_manager->num_rigid_contacts * 2] = glm::vec3(cpta.x, cpta.y, cpta.z);
  }
}

void ChOpenGLContacts::Update(ChSystem* physics_system) {
  contact_data.clear();
  if (ChSystemParallel* system_parallel = dynamic_cast<ChSystemParallel*>(physics_system)) {
    UpdateChronoParallel(system_parallel);
  } else {
    UpdateChrono(physics_system);
  }

  contacts.Update(contact_data);
}

void ChOpenGLContacts::TakeDown() {
  contacts.TakeDown();
  contact_data.clear();
}

void ChOpenGLContacts::Draw(const mat4& projection, const mat4& view) {
  glm::mat4 model(1);
  contacts.Draw(projection, view * model);
}
}
}
