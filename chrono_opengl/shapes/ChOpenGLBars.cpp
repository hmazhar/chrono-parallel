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
// Renders a wireframe view for triangles
// =============================================================================

#include <iostream>
#include "chrono_opengl/shapes/ChOpenGLBars.h"
#include <glm/gtc/type_ptr.hpp>

using namespace glm;
namespace chrono {
namespace opengl {

ChOpenGLBars::ChOpenGLBars() : ChOpenGLObject() {
}

bool ChOpenGLBars::Initialize(ChOpenGLShader* _shader) {
  if (this->GLReturnedError("ChOpenGLBars::Initialize - on entry"))
    return false;

  if (!super::Initialize()) {
    return false;
  }
  PostInitialize();

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), 0);  // Position

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  if (this->GLReturnedError("Cloud::Initialize - on exit"))
    return false;

  this->AttachShader(_shader);

  return true;
}

void ChOpenGLBars::AddBar(double left, double right, double top, double bottom, glm::vec3 color) {
  vec3 A(left, bottom, 0);
  vec3 B(left, top, 0);
  vec3 C(right, top, 0);
  vec3 D(right, bottom, 0);
  int index = this->data.size() - 1;

  this->data.push_back(ChOpenGLVertexAttributesPCN(A, color, glm::vec3(1, 0, 0)));
  this->data.push_back(ChOpenGLVertexAttributesPCN(B, color, glm::vec3(1, 0, 0)));
  this->data.push_back(ChOpenGLVertexAttributesPCN(C, color, glm::vec3(1, 0, 0)));
  this->data.push_back(ChOpenGLVertexAttributesPCN(D, color, glm::vec3(1, 0, 0)));

  this->vertex_indices.push_back(index + 0);
  this->vertex_indices.push_back(index + 1);
  this->vertex_indices.push_back(index + 2);
  this->vertex_indices.push_back(index + 3);

}

void ChOpenGLBars::Update() {
  this->data.clear();
  this->vertex_indices.clear();
}

bool ChOpenGLBars::PostInitialize() {
  std::size_t pcn_size = sizeof(ChOpenGLVertexAttributesPCN);

  if (this->GLReturnedError("ChOpenGLMesh::PostInitialize - on entry"))
    return false;
  // Generation complete bind everything!
  if (!this->PostGLInitialize((GLuint*)(&this->data[0]), this->data.size() * pcn_size)) {
    return false;
  }
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, pcn_size, (GLvoid*)(sizeof(vec3) * 0));  // Position
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, pcn_size, (GLvoid*)(sizeof(vec3) * 1));  // Color
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, pcn_size, (GLvoid*)(sizeof(vec3) * 2));  // Normal

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  if (this->GLReturnedError("ChOpenGLMesh::PostInitialize - on exit"))
    return false;

  return true;
}

void ChOpenGLBars::TakeDown() {
  data.clear();
  super::TakeDown();
}

void ChOpenGLBars::Draw(const mat4& projection, const mat4& view) {
  if (this->GLReturnedError("ChOpenGLCloud::Draw - on entry"))
    return;
  glEnable(GL_DEPTH_TEST);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // Enable the shader
  shader->Use();
  this->GLReturnedError("ChOpenGLCloud::Draw - after use");
  // Send our common uniforms to the shader
  shader->CommonSetup(value_ptr(projection), value_ptr(view));

  this->GLReturnedError("ChOpenGLCloud::Draw - after common setup");
  // Bind and draw! (in this case we draw as triangles)
  glBindVertexArray(this->vertex_array_handle);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_data_handle);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_element_handle);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertex_indices.size() * sizeof(GLuint), &vertex_indices[0], GL_DYNAMIC_DRAW);
  glDrawElements(GL_QUADS, this->vertex_indices.size(), GL_UNSIGNED_INT, (void*)0);
  this->GLReturnedError("ChOpenGLCloud::Draw - after draw");
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glUseProgram(0);

  if (this->GLReturnedError("ChOpenGLCloud::Draw - on exit"))
    return;
}
}
}
