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
// Generic renderable mesh. Based on code provided by Perry Kivolowitz.
// Authors: Hammad Mazhar
// =============================================================================

#include <iostream>
#include "ChOpenGLMesh.h"

using namespace std;
using namespace glm;
using namespace chrono::utils;

ChOpenGLMesh::ChOpenGLMesh()
      :
        ChOpenGLObject() {
}

bool ChOpenGLMesh::Initialize(
      std::vector<glm::vec3> &vertices,
      std::vector<glm::vec3> &normals,
      std::vector<glm::vec2> &texcoords,
      std::vector<GLuint> &indices,
      ChOpenGLMaterial mat) {
   if (this->GLReturnedError("Mesh::Initialize - on entry")) {
      return false;
   }

   if (!super::Initialize()) {
      return false;
   }

   this->position = vertices;
   this->normal = normals;
   this->texcoord = texcoords;
   this->vertex_indices = indices;

   for (unsigned int i = 0; i < vertices.size(); i++) {
      this->color_ambient.push_back(mat.ambient_color);
      this->color_diffuse.push_back(mat.diffuse_color);
      this->color_specular.push_back(mat.specular_color);

   }

   PostInitialize();

   if (this->GLReturnedError("ChOpenGLMesh::Initialize - on exit")) {
      return false;
   }

   return true;

}

bool ChOpenGLMesh::PostInitialize() {
   if (this->GLReturnedError("ChOpenGLMesh::PostInitialize - on entry"))
      return false;
   //Generation complete bind everything!
   if (!this->PostGLInitialize((GLuint*) (&this->position[0]), (GLuint*) (&this->normal[0]), (GLuint*) (&this->color_ambient[0]), (GLuint*) (&this->color_diffuse[0]),
                               (GLuint*) (&this->color_specular[0]), this->position.size() * sizeof(vec3))) {
      return false;
   }

   if (this->GLReturnedError("ChOpenGLMesh::PostInitialize - on exit"))
      return false;

   return true;
}
void ChOpenGLMesh::TakeDown() {
   //Clean up the vertex arrtibute data
   this->position.clear();
   this->normal.clear();
   this->color_ambient.clear();
   this->color_diffuse.clear();
   this->color_specular.clear();
   this->texcoord.clear();

   super::TakeDown();
}

/*  A note about the index arrays.

 In this example, the index arrays are unsigned ints. If you know
 for certain that the number of vertices will be small enough, you
 can change the index array type to shorts or bytes. This will have
 the two fold benefit of using less storage and transferring fewer
 bytes.
 */

void ChOpenGLMesh::Draw(
      const mat4 & projection,
      const mat4 & modelview) {
   if (this->GLReturnedError("ChOpenGLMesh::Draw - on entry"))
      return;

   glEnable(GL_DEPTH_TEST);
   //compute the mvp matrix and normal matricies
   mat4 mvp = projection * modelview;
   mat3 nm = inverse(transpose(mat3(modelview)));
   //bind any textures that we need

   //Enable the shader
   shader->Use();
   this->GLReturnedError("ChOpenGLMesh::Draw - after use");
   //Send our common uniforms to the shader
   shader->CommonSetup(value_ptr(projection), value_ptr(modelview), value_ptr(mvp), value_ptr(nm));

   this->GLReturnedError("ChOpenGLMesh::Draw - after common setup");
   //Bind and draw! (in this case we draw as triangles)
   glBindVertexArray(this->vertex_array_handle);

   glEnableVertexAttribArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_position_handle);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);  // Position

   glEnableVertexAttribArray(1);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_handle);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);  // Normal

   glEnableVertexAttribArray(2);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_ambient_handle);
   glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);  // Color Ambient

   glEnableVertexAttribArray(3);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_diffuse_handle);
   glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);  // Color Ambient

   glEnableVertexAttribArray(4);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_specular_handle);
   glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);  // Color Ambient

   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->vertex_element_handle);

   this->GLReturnedError("ChOpenGLMesh::Draw - before draw");
   glDrawElements(GL_TRIANGLES, this->vertex_indices.size(), GL_UNSIGNED_INT, (void*) 0);
   this->GLReturnedError("ChOpenGLMesh::Draw - after draw");
   //unbind everything and cleanup
   glDisableVertexAttribArray(0);
   glDisableVertexAttribArray(1);
   glDisableVertexAttribArray(2);
   glDisableVertexAttribArray(3);
   glDisableVertexAttribArray(4);
   glBindVertexArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   glUseProgram(0);

   if (this->GLReturnedError("ChOpenGLMesh::Draw - on exit"))
      return;
}