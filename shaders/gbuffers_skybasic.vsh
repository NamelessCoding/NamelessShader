#version 420

varying vec4 starData; //rgb = star color, a = flag for weather or not this pixel is a star.
varying vec4 PositionFinal;
void main() {

	gl_Position = ftransform();
		PositionFinal = gl_Position;

	starData = vec4(gl_Color.rgb, float(gl_Color.r == gl_Color.g && gl_Color.g == gl_Color.b && gl_Color.r > 0.0));
}