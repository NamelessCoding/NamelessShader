#version 450 compatibility
#extension GL_ARB_shading_language_packing: enable
varying vec2 TexCoords;


/* DRAWBUFFERS:4 */


// Direction of the sun (not normalized!)
uniform vec3 sunPosition;
uniform float viewHeight;
uniform float viewWidth;
uniform vec2 viewPixelSize; // = vec2(1.0 / viewWidth, 1.0 / viewHeight)

// The color textures which we wrote to
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex3;

uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;

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
float lum(vec3 c){
return 0.2126*c.x + 0.7152*c.y + 0.0722*c.z;

}

float g3x3(vec2 coords, vec2 iResolution){
vec2 p = coords * iResolution;
float ppp = 0.;
for(int i = 0; i < 9; i++){
vec2 coords2 = vec2(float(i%3)-1., float(i/3)-1.);
vec3 divideWith = texture2D(colortex7, (p+coords2)/iResolution).xyz;
ppp += lum(divideWith);
}
return ppp/9.;
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
float spll3 = texture2D(colortex5, TexCoords*0.5).w;
float mult = 32.;

vec2 currCoors = TexCoords;
vec3 currrp = texture2D(colortex4, TexCoords).xyz;
vec4 info = texture2D(colortex0, TexCoords);
vec4 info23;
if(spll3 >- 0.8){
info23 = texture2D(colortex9, TexCoords*0.5);
}
float currdepth = info.w;
vec3 currpos = info.xyz;
vec3 col = vec3(0.);
vec3 col2 = vec3(0.);
currCoors *= iResolution;
float spll = max(1.0-texture2D(colortex5, TexCoords*0.5).w, 0.0);

float check = texture2D(colortex7, TexCoords).w;
for(int i = 0; i < 25; i++){

vec2 coords = vec2(float(i%5)-2., float(i/5)-2.)*max(mult, 1.1);
vec2 fincords = (currCoors+coords)/iResolution;
float spll2 = max(1.0-texture2D(colortex5, fincords*0.5).w, 0.0);

vec3 nextrp = texture2D(colortex4, fincords).xyz;

vec4 info2 = texture2D(colortex0, fincords);
vec4 info24;
if(spll3 >- 0.8){
info24 = texture2D(colortex9, fincords*0.5);
}
float nextdepth = info2.w;
vec3 nextpos = info2.xyz;

float wd = exp(-(abs(min(currdepth, 100.) - min(nextdepth, 100.))/.645));
float wp = max(pow(max(dot(currpos, nextpos), 0.1), 128.), 0.01);
float wp2 = max(pow(max(dot(info23.xyz, info24.xyz), 0.1), 128.), 0.01);

//float wp = exp(-length(currpos - nextpos)/0.0285);
float wrp = exp(-(abs((1.0-lum(currrp)) - lum(nextrp))/(.75)));

float weigth = 1.0;
//if(check <= 1.0){
//weigth = wd*wp*spll;
//}else{
    float ws = exp(-abs(spll - spll2)/0.285);

weigth = max(wd,0.01)*wp*wrp*ws;
if(spll3 >= 0.8){
  weigth *= wp2;
}
weigth = max(weigth, 0.01);
//}
col += atrous_kernel_weights[i]*weigth*nextrp;
col2 += vec3(atrous_kernel_weights[i]*weigth);
}
return col/max(col2,0.001);
}

void main(){
    // Account for gamma correction


    vec2 iResolution = vec2(viewWidth, viewHeight);


   // //outColor = texture2D(colortex0, TexCoords).xyz;
//float mmm =  texture2D(colortex5, TexCoords*0.5).w;

    //outColor3 = vec4(col, 1.0);

  //  if(mmm != 20.5){
    gl_FragData[0] = vec4(blurAM(iResolution),texture2D(colortex4, TexCoords).w);
   // }


    //outColor3 = vec3(info.w*0.01);
   // outColor6 = info;
    
}