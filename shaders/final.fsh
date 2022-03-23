#version 450 compatibility
#extension GL_ARB_shading_language_packing: enable


varying vec2 TexCoords;
uniform sampler2D depthtex0;
uniform sampler2D noisetex;


uniform float viewHeight;
uniform float viewWidth;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex5;
uniform sampler2D colortex4;
uniform sampler2D colortex12;
uniform sampler2D colortex9;

uniform sampler2D colortex13;
#include "distort.glsl"

uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowModelViewInverse;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform float centerDepthSmooth;
uniform ivec2 eyeBrightnessSmooth;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
	uniform sampler2D colortex8;
uniform sampler2D shadowtex0; // Needed to enable shadow maps
uniform sampler2D shadowtex1; // Needed to enable shadow maps

uniform usampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform sampler2D colortex3;

uniform float near;
uniform float far;

//NOT MY CODE//////////////////////
vec3 ACESFilm(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 1.09;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.,1.);
}
float linearizeDepthFast(float depth) {

    return (near * far) / (depth * (near - far) + far);

}
//////////////////////////////////







vec3 blur3(vec2 p,float dist, vec2 iResolution){
p*=iResolution.xy;
    vec3 s;
    
    vec3 div = vec3(0.);
    //vec2 off = vec2(0.0, r);
    float k = 0.61803398875;
    for(int i = 0; i < 25; i++){
    vec2 coords = vec2(float(i%5)-2., float(i/5)-2.0)*dist;
    //vec3 c = texture2D(colortex7, vec2(p+coords)/iResolution.xy).xyz;
    vec2 cir = (p+coords)/iResolution.xy;
    vec3 c = (texture2D(colortex5, cir).xyz+ texture2D(colortex4, cir).xyz)*texture2D(colortex6, cir).xyz;

    //c = c*c *1.0;
    //vec3 bok = pow(c,vec3(4.));
      s+=c;
      //div += bok;
    }
        
    s/=25.;
    
    return s;
    
}

uniform vec3 sunPosition;


vec2 rot(vec2 a, float t){
float l = length(a);
a/=l;
float ang = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
ang += t*3.14159/180.;
return l*vec2(cos(ang), sin(ang));
}

vec3 pal(float t, vec3 a){
return 1.0 + 1.0*cos(2.*3.14159*t + a);
}

float torus(vec2 p, vec2 s){
    vec2 mm = normalize(p);
    return length(mm*s.x - p) - s.y;
}

float box(vec2 p, vec2 s){
vec2 a = abs(p)-s;
return max(a.x,a.y);
}

float hex(vec2 p, float s){
float box1 = box(p, vec2(s));
vec2 pos = rot(p, 45.);
return min(box1, box(pos, vec2(s)));
}

vec3 lens(vec2 p, vec2 mouse){
p*=10.;

vec3 col = vec3(0.);

col += sin(texture(noisetex, normalize(mouse-p)*0.7).x)*exp(-(length(mouse-p)-1.2)*1.);
col += exp(-(length(-mouse*1.0-p)-1.0)*3.);
for(int i = 0; i < 5; i++){
col += exp(-hex(-mouse*(0.2-float(i)*0.1)-p, 1.0-float(i)*0.1)*2.);
}

col += exp(-torus(mouse*0.2-p, vec2(2., 0.01))*2.);

vec3 col2 = vec3(1.)*exp(-torus(-mouse - p, vec2(6., 0.3))*20.);
col2 *= sin(texture(noisetex, normalize(-mouse-p)*0.7).x)*exp(-(length(-mouse-p)-2.2)*1.);
col2 *= vec3(1.0,1.0,0.2);
col2 *= length(mouse-p)*0.04;

vec3 col3 = vec3(1.)*exp(-torus(-mouse*0.2 - p, vec2(10., 2.))*4.);
col3 *= sin(texture(noisetex, normalize(-mouse-p)*0.7).x)*exp(-(length(-mouse-p)-2.2)*1.);
col3 *= vec3(1.0,1.0,0.2);
col3 *= length(mouse-p)*0.04;

col += col2;
col += col3;
col*=exp(-(length(mouse-p)-4.2)*0.2);
//col += exp(-(length(p)-4.));

col *= pal(length(mouse-p)*0.1, vec3(0.0,0.6,1.0));
col += exp(-(length(mouse-p)-0.6)*0.3)*vec3(1.0,0.7,0.2);

//vec2 pos2 = (mouse)-p;
//pos2 = rot(pos2, mouse.x*8.);
//col += exp(-length(pos2*vec2(1., 0.01)))*exp(-(length(mouse*0.2-p)-4.2)*0.6);

return col;
}

float lum(vec3 c){
return 0.2126*c.x + 0.7152*c.y + 0.0722*c.z;

}

float g3x3(vec2 coords, vec2 iResolution){
vec2 p = coords * iResolution;
float ppp = 0.;
for(int i = 0; i < 9; i++){
vec2 coords2 = vec2(float(i%3)-1., float(i/3)-1.);
float divideWith = texture2D(colortex4, (p+coords2)/iResolution).w;
ppp += divideWith;
}
return ppp/9.;
}

vec3 blurAM(in vec2 iResolution, sampler2D currusage, float mult = 2.0, vec2 curos){


//NOT MY CODE///////////////
const float atrous_kernel_weights[25] = {
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0 };
///////////////////////////
vec2 currCoors = curos;
vec3 currrp = texture2D(currusage, curos).xyz;
vec4 info = texture2D(colortex0, curos);
float currdepth = info.w;
vec3 currpos = info.xyz;
vec3 col = vec3(0.);
vec3 col2 = vec3(0.);
currCoors *= iResolution;
float check = texture2D(colortex7, curos).w;
float spll = max(1.0-texture2D(colortex3, curos).w, 0.0);
//if(spll < 0.5){return currrp;}
float spll3 = texture2D(colortex5, TexCoords).w;
if(spll3 == 250.){return currrp;}
for(int i = 0; i < 25; i++){
vec2 coords = vec2(float(i%5)-2., float(i/5)-2.)*max(mult, 1.1);
vec2 fincords = (currCoors+coords)/iResolution;
float spll2 = max(1.0-texture2D(colortex3, fincords).w, 0.0);

vec3 nextrp = texture2D(currusage, fincords).xyz;
vec3 nextrp2 = texture2D(currusage, fincords).xyz;

vec4 info2 = texture2D(colortex0, fincords);
float nextdepth = info2.w;
vec3 nextpos = info2.xyz;

float wd = exp(-(abs(currdepth - nextdepth)/.645));
float wp = pow(max(dot(currpos, nextpos), 0.0), 118.);
//float wp = exp(-length(currpos - nextpos)/0.0285);
float ws = exp(-abs(spll - spll2)/0.285);
float ddd = g3x3(fincords, iResolution);
float wrp = exp(-(abs((1.0-lum(currrp)) - lum(nextrp))/(sqrt(ddd)*.75 + 0.15)));
float weigth = 1.0;
//if(check <= 1.0){
//weigth = wd*wp*spll;
//}else{
weigth = wd*wp*ws*wrp;
//}
col += atrous_kernel_weights[i]*weigth*nextrp2;
col2 += vec3(atrous_kernel_weights[i]*weigth);

}
return col/max(col2,0.001);
}



vec3 blurt(vec2 coords2){
vec3 col = vec3(0.);
vec2 iResolution = vec2(viewWidth, viewHeight);
for(int i = 0; i < 9; i++){
vec2 coords = vec2(float(i&3)-1., float(i/3)-1.)*1.;
col += texture2D(colortex5, (coords2*iResolution + coords)/iResolution).xyz;
}
return col/9.;
}




//NOT MY CODE

float AdjustLightmapTorch(in float torch) {
    const float K = 2.0f;
    const float P = 5.06f;
    return K * pow(torch, P);
}

float AdjustLightmapSky(in float sky){
    float sky_2 = sky * sky;
    return sky_2 * sky_2;
}

vec2 AdjustLightmap(in vec2 Lightmap){
    vec2 NewLightMap;
    NewLightMap.x = AdjustLightmapTorch(Lightmap.x);
    NewLightMap.y = AdjustLightmapSky(Lightmap.y);
    return NewLightMap;
}

// Input is not adjusted lightmap coordinates
vec3 GetLightmapColor(in vec2 Lightmap){
    // First adjust the lightmap
    Lightmap = AdjustLightmap(Lightmap);
    // Color of the torch and sky. The sky color changes depending on time of day but I will ignore that for simplicity
    const vec3 TorchColor = vec3(1.0f, 0.25f, 0.08f);
    const vec3 SkyColor = vec3(0.05f, 0.15f, 0.3f);
    // Multiply each part of the light map with it's color
    vec3 TorchLighting = Lightmap.x * TorchColor;
    vec3 SkyLighting = Lightmap.y * SkyColor;
    // Add the lighting togther to get the total contribution of the lightmap the final color.
    vec3 LightmapLighting = TorchLighting + SkyLighting;
    // Return the value
    return LightmapLighting;
}

/////////////


vec3 blurt2(vec2 coords2){
vec3 col = vec3(0.);
vec2 iResolution = vec2(viewWidth, viewHeight);
for(int i = 0; i < 144; i++){
vec2 coords = vec2(float(i&12)-6., float(i/12)-6.)*6.;
col += GetLightmapColor(texture2D(colortex2, (coords2*iResolution + coords)/iResolution).xz);

}
return col/144.;
}

float blurt3(vec2 coords2){
float col = (0.);
vec2 iResolution = vec2(viewWidth, viewHeight);
for(int i = 0; i < 144; i++){
vec2 coords = vec2(float(i&12)-6., float(i/12)-6.)*1.;
col += texture2D(colortex6, (coords2*iResolution + coords)/iResolution).w;

}
return col/144.;
}

vec3 blur2(vec2 p,float dist, vec2 iResolution){
p*=iResolution.xy;
    vec3 s;
    
    vec3 div = vec3(0.);
    //vec2 off = vec2(0.0, r);
    float k = 0.61803398875;
    for(int i = 0; i < 150; i++){
    float m = float(i)*0.01;
    float r = 2.*3.14159*k*float(i);
    vec2 coords = vec2(m*cos(r), m*sin(r))*dist;
    //vec3 c = texture2D(colortex7, vec2(p+coords)/iResolution.xy).xyz;
    vec2 cir = (p+coords)/iResolution.xy;
    //float spll = max(texture2D(colortex5, TexCoords*1.0).w, 0.05);
float spll = max(texture2D(colortex5, cir*1.0).w, 0.05);

    //vec3 c = (blurAM(iResolution.xy, colortex5, 16.).xyz+ texture2D(colortex4, cir).xyz)*texture2D(colortex6, cir).xyz;
    //vec3 c = ((blurAM(iResolution,colortex5, 1.).xyz + blurAM(iResolution, colortex4)))*max(texture2D(colortex6,cir).xyz, spll);
//vec3 c = (max(blurt(cir)*2., 0.0) + texture2D(colortex4,cir).xyz + vec3(0.2,0.6,0.9)*0.0125 )*max(texture2D(colortex6,cir*1.0).xyz, spll);
//vec3 c = (blurt2(cir)+ max(blurt(cir)*2., 0.0) + blurAM(iResolution, colortex4, 2., cir).xyz + vec3(0.2,0.6,0.9)*0.0125 )*max(texture2D(colortex6,cir*1.0).xyz, spll);
vec3 c = (texture2D(colortex5, cir).xyz*4.0 + texture2D(colortex4, cir).xyz*2.)*texture2D(colortex6, cir).xyz;
//vec3 col = (max(blurt(TexCoords)*4., 0.0) + blurAM(iResolution, colortex4, 2., TexCoords)*3.)*texture2D(colortex6, TexCoords).xyz*1.;

    //c = c*c *1.0;
    vec3 bok = pow(c,vec3(4.));
      s+=c*bok;
      div += bok;
    }
        
    s/=div;
    
    return s;
    
}

vec3 bloom(vec2 p,float dist, vec2 iResolution){
p*=iResolution.xy;
    vec3 col = vec3(0.);
    for(int i = 0; i < 100; i++){
        vec2 coords = vec2(float(i%10)-5., float(i/10)-5.)*dist;
        vec2 fin = (TexCoords*iResolution+coords)/iResolution.xy;
          //  vec3 c = (texture2D(colortex5, fin).xyz+ texture2D(colortex4, fin).xyz)*texture2D(colortex6, fin).xyz;
   // vec3 c = (blurAM(iResolution.xy, colortex5, 1., fin).xyz+ blurAM(iResolution.xy, colortex4, 2., fin).xyz*texture2D(colortex6, fin).xyz);
//vec3 c = ( max(blurt(fin)*2., 0.0) + texture2D(colortex4, fin).xyz+ vec3(0.2,0.6,0.9)*0.0125 )*max(texture2D(colortex6,fin*1.0).xyz, 0.);
//vec3 c = (texture2D(colortex5, fin*0.5).xyz*4.0 + texture2D(colortex4, fin*0.5).xyz*2.)*texture2D(colortex3, fin*0.5).xyz;

vec3 alb = texture2D(colortex6, fin*0.5).xyz;
vec3 shadows = texture2D(colortex5, fin*0.5).xyz*2.0*alb;
vec3 indirect = texture2D(colortex4, fin).xyz*2.;
vec3 c = shadows + indirect*alb;

        col += pow(clamp(c,0.0,1.0), vec3(6.0));
    }
         col /= 25.;
    
    return col;
    
}

   /* for (int i = 0; i < 3; ++i){
        isVoxel = 0.0;
		
		gl_Position = vec4(gl_in[i].gl_Position.xyz,1.);

		gl_Position = shadowProjection * vec4(gl_Position.xyz,1.);
		gl_Position /= gl_Position.w;
		//gl_Position.xyz = gl_Position.xyz * 0.5 + 0.5;
		gl_Position.xy *= 2.;
        // If the triangle is outside the side dedicated to the shadowmap, don't emit the vertex
        if (gl_in[i].gl_Position.x < -1.0  ) {
            return;
    	}
        EmitVertex();
    }
    EndPrimitive();*/
    

vec3 quickShadow(){
    float Depth =  texture2D(colortex2, TexCoords).w;
    vec3 Position = texture2D(colortex2, TexCoords).xyz;

   //return Position;
//return Depth;
    ///NOT MY CODE////////////////////////////////////
    vec4 View = vec4(Position,1.);
 
    //return View.xyz;
    vec4 Projected = vec4(View.xyz, 1.);
   // vec4 Projected = vec4(View.xyz, 1.);
    Projected =  shadowProjection * shadowModelView * vec4(Projected.xyz, 1.);
  //  Projected =   *vec4(Projected.xyz, 1.);
   // Projected /= Projected.w;
    //Projected.xy = DistortPosition(Projected.xy);
    Projected.x += 0.5;
    Projected.xyz = Projected.xyz * 0.5 + 0.5;
    //Projected.x += 0.25;
    ////////////////////////////////////////////////////////
        //Projected.xy *= 2.;

    vec2 ProjectedCoordinates = Projected.xy;

    float readDepth = texture2D(shadowtex0, ProjectedCoordinates.xy).x;
    //return vec3(readDepth)*1.6;
//return linearizeDepthFast(Projected.z)*0.6;
    if(abs((readDepth) - (Projected.z)-0.001) > 0.005){
        return vec3(0.0);
    }
    return vec3(2.0);
}

/////NOT MY CODE///////////


vec3 uncharted2_tonemap_partial(vec3 x)
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 uncharted2_filmic(vec3 v)
{
    float exposure_bias = 2.0f;
    vec3 curr = uncharted2_tonemap_partial(v * exposure_bias);

    vec3 W = vec3(11.2f);
    vec3 white_scale = vec3(1.0f) / uncharted2_tonemap_partial(W);
    return curr * white_scale;
}

///////////////////////////

float saturation(vec3 rgb, float u){
return sqrt((pow(rgb.r - u,2.) + pow(rgb.g - u,2.) + pow(rgb.b - u,2.)  )/3.);

}

vec3 rgbtohsv(vec3 col){
float cmax = max(col.x, max(col.y, col.z));
float cmin = min(col.x, min(col.y, col.z));
float delta = cmax-cmin;
float h = 0.;
if(delta == 0.){
}else if(cmax == col.r){
h = 60.*mod((col.g - col.b)/delta, 6.);
}else if(cmax == col.g){
h = 60.*(((col.b - col.r)/delta) + 2.);
}else if(cmax == col.b){
h = 60.*(((col.r - col.g)/delta) + 4.);
}
h = clamp(h, 0., 360.);
float s = (cmax == 0.)?0.:delta/cmax;
s = clamp(s, 0., 1.);
float v = clamp(cmax,0.,1.);

return vec3(h,s,v);

}


vec3 hsvtorgb(vec3 hsv){
float C = hsv.z*hsv.y;
float hued = hsv.x / 60.;
float X = C * (1.0 - abs(mod(hued, 2.) - 1.));
vec3 rgb1 = vec3(0.);
if(hued <= 1. && hued >= 0.){
rgb1 = vec3(C, X, 0.);
}else if(hued <= 2. && hued >= 1.){
rgb1 = vec3(X, C, 0.);
}else if(hued <= 3. && hued >= 2.){
rgb1 = vec3(0., C, X);
}else if(hued <= 4. && hued >= 3.){
rgb1 = vec3(0., X, C);
}else if(hued <= 5. && hued >= 4.){
rgb1 = vec3(X, 0., C);
}else if(hued <= 6. && hued >= 5.){
rgb1 = vec3(C, 0., X);
}

float m = hsv.z - C;

return vec3(clamp(rgb1 + m, 0., 1.));

}

void main() {
    // Sample and apply gamma correction

    //float Depth = texture2D(depthtex0, TexCoords).r;
    //vec4 mcm = texture2D(colortex7, TexCoords*1.0).xyzw;
   // vec3 prev = texture2D(colortex0, TexCoords).xyz;
    //vec3 col = mcm.xyz*texture2D(colortex5, TexCoords*1.0).xyz;
                vec2 iResolution = vec2(viewWidth, viewHeight);

//vec3 col = blurAM(iResolution);
float spll = max(texture2D(colortex5, TexCoords*1.0).w, 0.05);
vec3 sunPos = mat3(gbufferModelViewInverse) * normalize(sunPosition);

//vec3 col = (max(blurAM(iResolution,colortex5, 1.).xyz, 0.1)*2.*max(dot(normalize(sunPos), texture2D(colortex0, TexCoords*1.0).xyz),0.4) 
//+ blurAM(iResolution, colortex4) + 0.0125 )*blurAM(iResolution, colortex4)*max(texture2D(colortex6,TexCoords*1.0).xyz, 0.)+blurAM(iResolution, colortex4)*0.2;
//col = GetLightmapColor(texture2D(colortex12, TexCoords).xy);
//    vec3 Diffuse = Albedo * (LightmapColor + NdotL * GetShadow(Depth) + Ambient);
//GetLightmapColor(texture2D(colortex2, TexCoords).xy)*0.125 + texture2D(colortex2, TexCoords).z*0.5+
//vec3 col = (blurt2(TexCoords) + max(blurt(TexCoords)*2., 0.0) + blurAM(iResolution, colortex4) + vec3(0.2,0.6,0.9)*0.0125 )*max(texture2D(colortex6,TexCoords*1.0).xyz, spll);

float deptish = texture2D(colortex2, TexCoords).w;
deptish = linearizeDepthFast(deptish);
//vec3 col = (blurt2(TexCoords)*0.125 + max(blurt(TexCoords)*2., 0.0) + blurAM(iResolution, colortex4) + vec3(0.2,0.6,0.9)*0.0125 )*max(texture2D(colortex6,TexCoords*1.0).xyz, spll);
//vec3 firstAlbedo = texture2D(colortex6,TexCoords*1.0).xyz;
//vec3 albedo = texture2D(colortex3, TexCoords).xyz;

//vec3 col = ( max(blurt(TexCoords)*2., 0.0) + blurAM(iResolution, colortex4, 2., TexCoords) )*max(firstAlbedo, spll);
//vec3 col = (max(blurt(TexCoords)*4., 0.0) + blurAM(iResolution, colortex4, 2., TexCoords)*5.)*alb*1.;
//vec3 col =  vec3(0.);

vec3 alb = texture2D(colortex6, TexCoords*0.5).xyz;
vec3 shadows = texture2D(colortex5, TexCoords*0.5).xyz*2.0*alb;
vec3 indirect = texture2D(colortex4, TexCoords).xyz*2.;

indirect = rgbtohsv(clamp(indirect,0.,1.));
//ol.z += 0.5;
//col.z = clamp(col.z, 0., 1.);
indirect.y *= 2.;
indirect.y = clamp(indirect.y, 0., 1.);
indirect = hsvtorgb(indirect);

vec3 col = (shadows + indirect*alb);

//col = rgbtohsv(clamp(col,0.,1.));
//col.z += 0.5;
//col.z = clamp(col.z, 0., 1.);
//col.y *= 2.;//
//col.y = clamp(col.y, 0., 1.);
//col = hsvtorgb(col);

//vec3 col = (max(blurt()*2., 0.0) + blurAM(iResolution, colortex4))*max(texture2D(colortex6,TexCoords*1.0).xyz, spll);

//vec3 col = texture2D(colortex7, TexCoords*1.0).xyz;

float mmm =  texture2D(colortex5, TexCoords*0.5).w;

    //outColor3 = vec4(col, 1.0);
    vec4 info = texture2D(colortex6, TexCoords*0.5);

    if(mmm == 20.5){
        col = vec3(0.);
        for(int i = 0; i < 9; i++){

        vec2 coords = vec2(float(i%3)-1., float(i/3)-1.)*1.;
        vec2 fin = (TexCoords*iResolution+coords)/iResolution.xy;
        col += texture2D(colortex5, fin*0.5).xyz;
        }
         col /= 9.;
col = pow(col, vec3(1.2));
       col = ACESFilm(col);
        col = pow(col, vec3(1./2.2));
         
  }else{
float dist = info.w;
    dist *= 0.1;
    float depthcenter = texture2D(colortex6, vec2(0.5)).w*0.1;
    //col = blur2(TexCoords*1.0, clamp(dist*1.2*(7.-clamp(depthcenter*12.1, 0., 7.)),0.0,5.), iResolution);


   float f = clamp(exp(-dist*dist*0.001), 0.6, 1.);
    col = col*f + (1.0-f)*vec3(0.6,0.7,0.9);
//col = vec3(linearizeDepthFast(centerDepthSmooth*0.1, 0.1, 100.));
//col = texture2D(colortex7, TexCoords*1.0).xyz;

  }
//col = GetLightmapColor(texture2D(colortex2, TexCoords).xy);
//col = vec3(quickShadow())*texture2D(colortex3, TexCoords).xyz;
//col += vec3(0.9, 0.6, 0.2)*blurt3(TexCoords)*0.5;
//col = texture2D(colortex4, TexCoords).xyz;
//col = GetLightmapColor(texture2D(colortex13, TexCoords).xy);
//uvec2 data = uvec2(texelFetch(shadowcolor0, ivec2(TexCoords*vec2(2048.)), 0).xy);
//col = linearizeDepthFast(unpackUnorm4x8(data.x).w, 0.1, 100.);

		//return vec4[2](unpackUnorm4x8(data.x), unpackUnorm4x8(data.y));
col = vec3(1.)-exp(-1.9*col);
///col += lens(TexCoords*2.0-1.0, sunPosition.xy*0.7)*0.4;
col += bloom(TexCoords*0.5, 1.0, iResolution)*0.1;
    //NOT MY CODE//////////////////
        col = vec3(1.)-exp(-1.*col);
vec3 a = vec3(1.3,.6,.3)-0.4;
    col = mix(col, smoothstep(0.,1.,col),a);
   // vec3 aa = vec3(1.2,1.6,1.9);
   // col = sqrt(col/aa);
  /*  
  vec3 a = vec3(0.6,1.0,1.5)-0.4;
    col = mix(col, smoothstep(0.,1.,col),a);
    vec3 aa = vec3(1.9,1.8,1.7);
    col = sqrt(col/aa);
  vec3 a = vec3(0.6,1.0,1.0)-0.4;
    col = mix(col, smoothstep(0.,1.,col),a);
    //////////////////////////////
    // Output to screen
    vec3 aa = vec3(1.9,1.8,1.7);
    col = sqrt(col/aa);
    col = (1./((1.)+exp(-(10.)*(col-0.5))));
    //col = ACESFilm(col);
    //col = pow(col, vec3(1./2.2));
    //col *= exp(-length(uv*2.0-1.0)*0.6);
    // Output to screen*/
    //col += vec3(saturation(col, 0.5));
    //col = pow(col, vec3(1.6));
    //col = uncharted2_filmic(col);
    col = ACESFilm(col);
   col = pow(col, vec3(1./2.2));

  gl_FragColor = vec4(col, 1.0f);
}