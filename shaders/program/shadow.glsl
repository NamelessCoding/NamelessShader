//--// Settings
//Modified code from Lith https://cdn.discordapp.com/attachments/276979724922781697/862415868934619206/Voxelization_Geometry.zip


#define VOXELIZATION_PASS

//--// Uniforms

uniform vec3 cameraPosition;

uniform mat4 shadowModelViewInverse;

uniform sampler2D tex;
uniform sampler2D normals;
uniform sampler2D colortex2;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;

uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

#if STAGE == STAGE_VERTEX
//--// Vertex Inputs

#define attribute in
attribute vec3 at_midBlock;
attribute vec2 mc_Entity;
attribute vec2 mc_midTexCoord;

//--// Vertex Outputs

out vec4 tint;
out vec3 scenePosition;
out vec3 worldNormal;
out vec3 midBlock;
out vec2 texCoord;
out vec2 midCoord;
out int blockId;
out float isNotOpaque;

//--// Vertex Functions

void main() {
  gl_Position = shadowModelView * gl_Vertex;

  tint = gl_Color;
  scenePosition = (shadowModelViewInverse * gl_Position).xyz;
  worldNormal =
      normalize(mat3(shadowModelViewInverse) * gl_NormalMatrix * gl_Normal);
  texCoord = gl_MultiTexCoord0.st;
  midBlock = (at_midBlock / 64.0);
  midCoord = mc_midTexCoord;
  blockId = int(mc_Entity.x);
  //	gl_Position = ftransform();

  if (blockId == 0) {
    isNotOpaque = 1.0;
  } else {
    isNotOpaque = 0.0;
    blockId = max(blockId, 1);
  }
}
#elif STAGE == STAGE_GEOMETRY
//--// Geometry Inputs

layout(triangles) in;

in vec4[3] tint;
in vec3[3] scenePosition;
in vec3[3] worldNormal;
in vec3[3] midBlock;
in vec2[3] texCoord;
in vec2[3] midCoord;
in int[3] blockId;
in float[3] isNotOpaque;

//--// Geometry Outputs

layout(triangle_strip, max_vertices = 7) out;

out vec4 fData0;
out vec4 fData1;
out float isVoxel;
//--// Geometry Libraries

#include "/lib/voxelization.glsl"

//--// Geometry Functionss

float maxof(vec2 x) { return max(x.x, x.y); }
float maxof(vec3 x) { return max(x.x, max(x.y, x.z)); }
float minof(vec3 x) { return min(x.x, min(x.y, x.z)); }

vec2 rot(vec2 a, float b) {
  float l = length(a);
  a = normalize(a);
  float ang = (a.y < 0.) ? 2. * 3.14159 - acos(a.x) : acos(a.x);
  ang += b;
  return l * vec2(cos(ang), sin(ang));
}

void main() {
  for (int i = 0; i < 3; ++i) {
    isVoxel = 0.0;

    gl_Position = vec4(gl_in[i].gl_Position.xyz, 1.);
    // gl_Position.xyz = gl_Position.xyz * 2. - 1.;
    // gl_Position.x += 1.0;

    gl_Position = shadowProjection * vec4(gl_Position.xyz, 1.);
    // gl_Position.xy = rot(gl_Position.xy, 1.);

    // gl_Position /= gl_Position.w;
    // gl_Position.xyz = gl_Position.xyz * 0.5 + 0.5;
    gl_Position.x += 0.5;

    // If the triangle is outside the side dedicated to the shadowmap, don't
    // emit the vertex
    if (gl_Position.x < -0.0) {
      return;
    }
    EmitVertex();
  }
  EndPrimitive();
  if (isNotOpaque[0] > 0.5) {
    return;
  }

  vec3 triCentroid =
      (scenePosition[0] + scenePosition[1] + scenePosition[2]) / 3.0;
  vec3 midCentroid = (midBlock[0] + midBlock[1] + midBlock[2]) / 3.0;

  // voxel position in the 2d map
  vec3 voxelSpacePosition = SceneSpaceToVoxelSpace(triCentroid + midCentroid);
  ivec3 voxelIndex = ivec3(floor(voxelSpacePosition));
  if (!IsInVoxelizationVolume(voxelIndex)) {
    return;
  }
  vec4 p2d = vec4(
      ((GetVoxelStoragePos(voxelIndex) + 0.5) / float(shadowMapResolution)) *
              2.0 -
          1.0,
      worldNormal[0].y * -0.25 + 0.5, 1.0);

  // fill out data
  ivec2 atlasResolution = textureSize(tex, 0);
  vec2 atlasAspectCorrect =
      vec2(1.0, float(atlasResolution.x) / float(atlasResolution.y));
  float tileSize = maxof(abs(texCoord[0] - midCoord[0]) / atlasAspectCorrect) /
                   maxof(abs(scenePosition[0] - scenePosition[1]));
  vec2 tileOffset =
      round((midCoord[0] - tileSize * atlasAspectCorrect) * atlasResolution);
  tileSize = round(2.0 * tileSize * atlasResolution.x);
  tileOffset = round(tileOffset / tileSize);

  vec4[2] voxel = vec4[2](0.0);
  SetVoxelTint(voxel, tint[0].rgb);
  SetVoxelId(voxel, blockId[0]);
  SetVoxelTileSize(voxel, int(tileSize));
  SetVoxelTileIndex(voxel, ivec2(tileOffset));

  // Create the primitive
  fData0 = voxel[0];
  fData1 = voxel[1];

  const vec2[4] offs =
      vec2[4](vec2(-1, 1), vec2(1, 1), vec2(1, -1), vec2(-1, -1));
  for (int i = 0; i < 4; ++i) {
    isVoxel = 1.0;
    gl_Position = p2d;
    gl_Position.xy += offs[i] / int(MC_SHADOW_QUALITY * shadowMapResolution);
    EmitVertex();
  }
  EndPrimitive();
}
#elif STAGE == STAGE_FRAGMENT
//--// Fragment Inputs

in vec4 fData0;
in vec4 fData1;
in float isVoxel;

in vec2 texCoord;
in vec4 tint;
uniform sampler2D texture;

//--// Fragment Functionss

float linearizeDepthFast(float depth, float near, float far) {
  return (near * far) / (depth * (near - far) + far);
}

/* RENDERTARGETS: 0 */
layout(location = 0) out uvec2 voxelData;

void main() {
  if (isVoxel > 0.5) {
    voxelData = uvec2(packUnorm4x8(fData0), packUnorm4x8(fData1));
  } else {
    voxelData =
        uvec2(packUnorm4x8(texture2D(texture, texCoord).xyzw * tint), uint(20));
    // gl_FragData[1] = vec4(texture2D(texture, texCoord).xyzw*tint);
  }

  // empty = vec4(1., 0., 1., 1.);
}
#endif
