#version 450 compatibility
#extension GL_ARB_shading_language_packing: enable
varying vec2 TexCoords;

/* DRAWBUFFER : 4 */

// Direction of the sun (not normalized!)
uniform vec3 sunPosition;
uniform float viewHeight;
uniform float viewWidth;
uniform vec2 viewPixelSize; // = vec2(1.0 / viewWidth, 1.0 / viewHeight)

// The color textures which we wrote to
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex5;
uniform sampler2D colortex4;

uniform sampler2D depthtex0;
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D noisetex;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;


float linearizeDepthFast(float depth, float near, float far) {

    return (near * far) / (depth * (near - far) + far);

}

vec3 blurAM(in vec2 iResolution){

//NOT MY CODE///////////////
const float atrous_kernel_weights[25] = {
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0 };
///////////////////////////
vec2 currCoors = TexCoords;
vec3 currrp = texture2D(colortex4, TexCoords).xyz;
vec4 info = texture2D(colortex0, TexCoords);
float currdepth = info.w;
vec3 currpos = info.xyz;
vec3 col = vec3(0.);
vec3 col2 = vec3(0.);
currCoors *= iResolution;
for(int i = 0; i < 25; i++){
float spll = max(1.0-texture2D(colortex5, TexCoords).w, 0.05);
vec2 coords = vec2(float(i%5)-2., float(i/5)-2.)*max(2.0, 2.0);
vec2 fincords = (currCoors+coords)/iResolution;

vec3 nextrp = texture2D(colortex4, fincords).xyz;
vec4 info2 = texture2D(colortex0, fincords);
float nextdepth = info2.w;
vec3 nextpos = info2.xyz;

float wd = exp(-(abs(currdepth - nextdepth)/.45));
//loat wp = pow(max(dot(currpos, nextpos), 0.0), 1.);
float wp = exp(-length(currpos - nextpos)/0.285);

float wrp = exp(-(length(currrp - nextrp)/.45));
float weigth = wd*wp*spll;
col += atrous_kernel_weights[i]*weigth*nextrp;
col2 += vec3(atrous_kernel_weights[i]*weigth);

}
return col/max(col2,0.01);
}

void main(){
     vec2 iResolution = vec2(viewWidth, viewHeight);

    //outColor3 = blurAM(iResolution);
float mmm =  texture2D(colortex5, TexCoords).w;

    //outColor3 = vec4(col, 1.0);

  if(mmm != 20.5){
   gl_FragData[4] = vec4(blurAM(iResolution),1.0f);
     //   gl_FragData[4] = vec4(texture2D(colortex4,TexCoords).xyz,1.0f);

    }

 

    //outColor = texture2D(colortex0, TexCoords).xyz;
   // outColor3 = blurAM(iResolution);
    
}