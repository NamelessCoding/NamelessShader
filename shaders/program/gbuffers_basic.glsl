
#if     STAGE == STAGE_VERTEX
varying vec2 lmcoord;
varying vec4 glcolor;

#define attribute in
	attribute vec3 at_midBlock;
	attribute vec2 mc_Entity;
	attribute vec2 mc_midTexCoord;


out vec2 texcoord;
out vec3 normal;
out vec3 pos;
out float blockId;

void main() {
	gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * gl_Vertex);
	pos = vec3(gl_Vertex.xyz);
	texcoord = gl_MultiTexCoord0.st;
	normal = gl_NormalMatrix * gl_Normal;
	lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
	blockId = (mc_Entity.x);

	glcolor = gl_Color;
}

#elif STAGE == STAGE_FRAGMENT

in vec2 texcoord;
in vec3 normal;
in vec3 vid;
in vec3 pos;
in float blockId;
uniform sampler2D lightmap;

varying vec2 lmcoord;
varying vec4 glcolor;

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;
uniform mat4 gbufferModelViewInverse;
uniform float viewWidth;
uniform float viewHeight;
uniform sampler2D depthtex2;

//////NOT MY CODE//////////////
// Creates a TBN matrix from a normal and a tangent

mat3 tbnNormalTangent(vec3 normal, vec3 tangent) {
    vec3 bitangent = cross(normal, tangent);
    return mat3(tangent, bitangent, normal);
}

// Creates a TBN matrix from just a normal
// The tangent version is needed for normal mapping because
//   of face rotation

mat3 tbnNormal(vec3 normal) {
    // This could be
    // normalize(vec3(normal.y - normal.z, -normal.x, normal.x))
    vec3 tangent = normalize(cross(normal, vec3(0, 1, 1)));
    return tbnNormalTangent(normal, tangent);
}
///////////////////////////////


/* DRAWBUFFERS:023 */

/* RENDERTARGETS: 0,2,3 */
	layout(location = 0) out vec4 data0;
	layout(location = 1) out vec4 data2;
	layout(location = 2) out vec4 data3;

void main() {
    if(texture2D(texture, texcoord).a < 0.102) discard;

	vec4 color = glcolor;
	vec3 fn = (texture2D(normals, texcoord).xyz);
vec3 nn = mat3(gbufferModelViewInverse) * normal;

	fn.xy = fn.xy * 2.0 - 1.0;
	fn.z  = sqrt(1.0 - dot(fn.xy, fn.xy));
	//fn    = TBN * fn;
  mat3 tbn = tbnNormal(mat3(gbufferModelViewInverse) * normal);
  
	//color *= texture2D(lightmap, lmcoord);
//mat3(gbufferModelViewInverse) * 
//vec3 tbn = tbnNormal();
	data0 = vec4(normalize(tbn*fn), 1.);
	data2 = vec4(pos,gl_FragCoord.z);
	data3 = vec4(pow(texture2D(texture, texcoord).xyz*color.xyz, vec3(2.2)), texture2D(specular, texcoord).x);

}

#endif
