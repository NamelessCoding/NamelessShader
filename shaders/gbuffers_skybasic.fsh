#version 420

uniform float viewHeight;
uniform float viewWidth;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;

uniform mat4 gbufferModelViewInverse;

uniform mat4 gbufferProjectionInverse;
uniform vec3 fogColor;
uniform vec3 skyColor;

varying vec4 starData; //rgb = star color, a = flag for weather or not this pixel is a star.
varying vec4 PositionFinal;
uniform vec3 cameraPosition;





void main() {
	vec3 color = vec3(1.0);
	
/* DRAWBUFFERS:0 */
	gl_FragData[0] = vec4(color, 1.0); //gcolor
}