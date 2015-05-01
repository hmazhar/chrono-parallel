#version 330

layout(location = 0) out vec4 FragColor;
uniform vec4 color;
in vec3 pos;
uniform float point_size;
uniform mat4 projection_matrix;
uniform mat4 view_matrix;
const vec3 lightPos = vec3(500, 500, 500);
const vec3 lightDir = normalize(lightPos);


void main() {
  //calculate normal
  vec3 normal;
  normal.xy = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(normal.xy, normal.xy);
  
  if (r2 > 1.0) {
    discard;
  }
 
  normal.z = sqrt(1.0 - r2);

  float diffuse = max(.3, dot(lightDir, normal));
  FragColor = vec4(color.xyz *diffuse, 1);

  //calculate depth
  vec4 pixelPos = vec4(pos + normal * point_size, 1.0);
  vec4 clipSpacePos = projection_matrix * pixelPos;
  
  gl_FragDepth = (clipSpacePos.z / clipSpacePos.w) * 0.5f + 0.5f;
}
