#version 450 compatibility
#extension GL_ARB_shading_language_packing : enable
varying vec2 TexCoords;

/*
const int colortex13Format = RGBA32F;
const bool colortex13Clear = false;
const bool colortex13MipmapEnabled = false;
const vec4 colortex13ClearColor = vec4(1,1,1,1);
*/

/* RENDERTARGETS: 13 */
layout(location = 0) out vec4 data13;

// Direction of the sun (not normalized!)
uniform vec3 sunPosition;
uniform float viewHeight;
uniform float viewWidth;
uniform vec2 viewPixelSize;  // = vec2(1.0 / viewWidth, 1.0 / viewHeight)

// The color textures which we wrote to
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex5;
uniform sampler2D colortex4;
uniform sampler2D colortex13;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D depthtex2;
uniform usampler2D shadowcolor0;
uniform usampler2D shadowcolor1;
uniform sampler2D shadowtex0;
uniform sampler2D noisetex;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform int frameCounter;

uniform float near;
uniform float far;
//Modified code from Lith https://cdn.discordapp.com/attachments/276979724922781697/862415868934619206/Voxelization_Geometry.zip

#include "/lib/voxelization.glsl"
#include "/lib/raytracer.glsl"

struct Material {
  vec3 albedo;  // Scattering albedo
  float opacity;
  vec3 normal;  // Normal
  float spec;
};

Material getMaterial(vec4 albedo, vec4 specular, vec4 normal, int voxelID) {
  Material material;

  material.albedo = albedo.rgb;
  material.opacity = albedo.a;
  material.spec = specular.x;
  material.normal = normal.rgb;

  return material;
}

Material getMaterial(vec3 position, vec3 normal, vec4[2] voxel) {
  int id = ExtractVoxelId(voxel);

  // Figure out texture coordinates
  int tileSize = ExtractVoxelTileSize(voxel);
  ivec2 tileOffs = ExtractVoxelTileIndex(voxel) * tileSize;

  ivec2 tileTexel;
  mat3 tbn;

  if (abs(normal.y) > abs(normal.x) && abs(normal.y) > abs(normal.z)) {
	tileTexel = ivec2(fract(position.x) * tileSize,
					  fract(position.z * sign(normal.y)) * tileSize);
	tbn = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, sign(normal.y)), normal);
  } else {
	tileTexel =
		ivec2(fract(position.x * sign(normal.z) - position.z * sign(normal.x)) *
				  tileSize,
			  fract(-position.y) * tileSize);
	tbn = mat3(vec3(sign(normal.z), 0.0, -sign(normal.x)), vec3(0.0, -1.0, 0.0),
			   normal);
  }

  ivec2 texelIndex = tileOffs + tileTexel;

  // Read textures
  vec4 baseTex = texelFetch(depthtex0, texelIndex, 0);
  vec4 specularTex = texelFetch(depthtex1, texelIndex, 0);
  vec4 normalTex = texelFetch(depthtex2, texelIndex, 0);

  normalTex.xyz = normalTex.xyz * 2.0 - 1.0;
  normalTex.xyz = normalize(tbn * normalTex.xyz);

  baseTex.rgb *= ExtractVoxelTint(voxel);

  return getMaterial(baseTex, specularTex, normalTex, id);
}

bool IntersectVoxelAlphatest(vec3 origin, ivec3 index, vec3 direction,
							 vec4[2] voxel, int id, out vec3 hitPos,
							 inout vec3 hitNormal) {
  if (IntersectVoxel(origin, index, direction, id, hitPos, hitNormal)) {
	// Perform alpha test
	Material voxelMaterial = getMaterial(hitPos, hitNormal, voxel);
	return (voxelMaterial.opacity > 0.102);
  }

  return false;
}

bool RaytraceVoxel(vec3 origin, ivec3 originIndex, vec3 direction,
				   bool transmit, const int maxSteps, out vec4[2] voxel,
				   out vec3 hitPos, out ivec3 hitIndex, out vec3 hitNormal,
				   out int id) {
  voxel = ReadVoxel(originIndex);
  hitIndex = originIndex;

  id = ExtractVoxelId(voxel);
  if (id > 0 && !transmit) {
	// bool IntersectVoxelAlphatest(vec3 origin, ivec3 index, vec3 direction,
	// int id, out vec3 hitPos, inout vec3 hitNormal) {

	if (IntersectVoxelAlphatest(origin, originIndex, direction, voxel, id,
								hitPos, hitNormal)) {
	  return true;
	}
  }

  vec3 deltaDist;
  vec3 next;
  ivec3 deltaSign;
  for (int axis = 0; axis < 3; ++axis) {
	deltaDist[axis] = length(direction / direction[axis]);
	if (direction[axis] < 0.0) {
	  deltaSign[axis] = -1;
	  next[axis] = (origin[axis] - hitIndex[axis]) * deltaDist[axis];
	} else {
	  deltaSign[axis] = 1;
	  next[axis] = (hitIndex[axis] + 1.0 - origin[axis]) * deltaDist[axis];
	}
  }

  bool hit = false;

  for (int i = 0; i < maxSteps && !hit; ++i) {
	if (next.x > next.y) {
	  if (next.y > next.z) {
		next.z += deltaDist.z;
		hitIndex.z += deltaSign.z;
	  } else {
		next.y += deltaDist.y;
		hitIndex.y += deltaSign.y;
	  }
	} else if (next.x > next.z) {
	  next.z += deltaDist.z;
	  hitIndex.z += deltaSign.z;
	} else {
	  next.x += deltaDist.x;
	  hitIndex.x += deltaSign.x;
	}

	if (!IsInVoxelizationVolume(hitIndex)) {
	  break;
	}

	voxel = ReadVoxel(hitIndex);
	id = ExtractVoxelId(voxel);
	if (id > 0) {
	  hit = IntersectVoxelAlphatest(origin, hitIndex, direction, voxel, id,
									hitPos, hitNormal);
	}
  }

  return hit;
}

bool rayTrace(vec3 dir, vec3 origin, out vec3 hitPos, out vec3 hitNormal,
			  out vec4[2] voxel, out int id, int stepssize) {
  // ec3 startVoxelPosition =
  // SceneSpaceToVoxelSpace(gbufferModelViewInverse[3].xyz);

  ivec3 hitIdx;
  bool hit = RaytraceVoxel(origin, ivec3(floor(origin)), dir, true, stepssize,
						   voxel, hitPos, hitIdx, hitNormal, id);
  return hit;
}
/***********************************************************************/

float linearizeDepthFast(float depth, float near, float far) {
  return (near * far) / (depth * (near - far) + far);
}
//////////////////////////////////
// / Code from: https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
uint wang_hash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

float rndf(inout uint state) { return float(wang_hash(state)) / 4294967296.0; }
////////////////////////

vec3 cosineweighted(vec3 d, inout uint r) {
  float r1 = rndf(r);
  float r2 = rndf(r);

  float x = cos(r1 * 3.14159 * 2.) * sqrt(1.0 - r2);
  float y = sin(r1 * 3.14159 * 2.) * sqrt(1.0 - r2);
  float z = sqrt(r2);

  vec3 N = d;
  vec3 W = (abs(N.x) > 0.99) ? vec3(0., 1., 0.) : vec3(1., 0., 0.);
  vec3 T = normalize(cross(N, W));
  vec3 B = normalize(cross(N, T));

  return normalize(T * x + B * y + z * N);
}

vec3 DoIT(inout uint r) {
  vec2 iResolution = vec2(viewWidth, viewHeight);

  vec4[2] voxel;
  // vec3 n = texture2D(colortex0, TexCoords).xyz;
  vec3 n = texelFetch(colortex0, ivec2(TexCoords * iResolution), 0).xyz;
  // return n;
  vec3 p = SceneSpaceToVoxelSpace(texture2D(colortex2, TexCoords * 0.5).xyz);
  p += n * 0.05;
  // return p;
  vec3 d = cosineweighted(n.xzy, r).xzy;
  // return d;
  vec3 origin = p;
  int ID = 0;
  bool hit = rayTrace(d, origin, p, n, voxel, ID, 258);
  vec3 c = getMaterial(p, n, voxel).albedo;

  vec3 final =
	  texelFetch(colortex13, ivec2(TexCoords * iResolution * 0.125), 0).xyz;
  if (hit) {
	if (ID == 58 || ID == 59 || ID == 60 || ID == 61 || ID == 62 || ID == 63 ||
		ID == 64 || ID == 65 || ID == 66 || ID == 67 || ID == 68 || ID == 69 ||
		ID == 70 || ID == 45 || ID == 46 || ID == 47 || ID == 48 || ID == 49 ||
		ID == 50 || ID == 51 || ID == 52 || ID == 53 || ID == 54 || ID == 55 ||
		ID == 56 || ID == 300 || ID == 57) {
	  return p;
	}
  }
  return (final);
}

void main() {
  uint r = uint(uint(TexCoords.x * 1000.) * uint(1973) +
				uint(TexCoords.y * 1000.) * uint(9277) +
				uint(frameCounter) * uint(26699)) |
		   uint(1);
  // inout uint r

  data13 = vec4(DoIT(r), 1.);
}