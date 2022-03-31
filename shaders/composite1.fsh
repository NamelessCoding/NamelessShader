#version 450 compatibility
#extension GL_ARB_shading_language_packing: enable





/* RENDERTARGETS: 0,2,4,7,11 */
	layout(location = 0) out vec4 data0;
	layout(location = 1) out vec4 data2;
	layout(location = 2) out vec4 data4;
	layout(location = 3) out vec4 data7;
	layout(location = 4) out vec4 data11;

varying vec2 TexCoords;

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
uniform sampler2D colortex5;
uniform sampler2D colortex4;
uniform sampler2D colortex9;
uniform sampler2D colortex10;
uniform sampler2D colortex11;
    uniform sampler2D colortex13;

uniform int frameCounter;


uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex12;

uniform sampler2D depthtex0;
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D noisetex;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;

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

//NOT MY CODE////////////
uint wang_hash(inout uint seed)
{
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}
 
float rndf(inout uint state)
{
    return float(wang_hash(state)) / 4294967296.0;
}
////////////////////////


//MY CODE
float lessthanone(float x, float B, float C){
return ((12. - 9.*B - 6.*C)*abs(x*x*x) + (-18.+12.*B+6.*C)*abs(x*x) + (6.-2.*B))/6.;
}

float higherthan1and2(float x, float B, float C){
return ((-B-6.*C)*abs(x*x*x) + (6.*B + 30.*C)*abs(x*x) + (-12.*B - 48.*C)*abs(x) + (8.*B+24.*C))/6.;
}
//
float MN(float dist, float B, float C){
float have = 0.;
if(dist <= 1.0){
    have = lessthanone(dist, B, C);
}else{
    have = higherthan1and2(dist, B, C);
}
return have;
}

vec3 cosineweighted(vec3 d, inout uint r){
float r1 = rndf(r);
float r2 = rndf(r);

float x = cos(r1*3.14159*2.)*sqrt(1.0-r2);
float y = sin(r1*3.14159*2.)*sqrt(1.0-r2);
float z = sqrt(r2);

vec3 N = d;
vec3 W = (abs(N.x) > 0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
vec3 T = normalize(cross(N,W));
vec3 B = normalize(cross(N,T));

return normalize(T*x + B*y + z*N);
}

void main(){
    // Account for gamma correction
    vec3 eyeCam = cameraPosition + gbufferModelViewInverse[3].xyz;
    vec4 normalsAndDepth = texture2D(colortex0, TexCoords*0.5);
float CheckIfSky =  texture2D(colortex5, TexCoords*0.5).w;
float Depth =  texture2D(colortex2, TexCoords*0.5).w;

uint r = uint(uint(TexCoords.x*1000.) * uint(1973) 
    + uint(TexCoords.y * 1000.) * uint(9277) 
    + uint(frameCounter) * uint(26699)) | uint(1);

    ///NOT MY CODE////////////////////////////////////
    vec4 View = vec4((TexCoords)*2.0-1.0, Depth*2.0-1.0, 1.);

    View = gbufferProjectionInverse*View;
    View /= View.w;
    View = gbufferModelViewInverse*View ;

    vec3 cameraOffset =  cameraPosition-previousCameraPosition;

    vec4 Projected = vec4(View.xyz, 1.) + vec4(cameraOffset, 0.);
    Projected =  gbufferPreviousModelView*Projected;
    Projected =   gbufferPreviousProjection*Projected;
    Projected /= Projected.w;

    ////////////////////////////////////////////////////////
    vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
    
    float velocity = length(ProjectedCoordinates - TexCoords)*1.8;
    vec2 Resolution = vec2(viewWidth, viewHeight);
    velocity = exp(-length((ProjectedCoordinates - TexCoords)*Resolution*0.01))*1. ;


    //vec4 infoprev = texture2D(colortex3, ProjectedCoordinates.xy);
    float depth1 =  texture2D(colortex6, TexCoords*0.5).w;
    float depth2 =  texture2D(colortex4, ProjectedCoordinates).w;

    float weightDepth = exp(-(abs(depth1 - depth2)/1.38));
    //float wp = exp(-(length(info.xyz - infoprev.xyz)/2.28));
    float finalWeigth = weightDepth*velocity;


    vec3 norm = normalsAndDepth.xyz;
    vec3 pos = View.xyz + cameraOffset;
    float ambientO = 0.;
    for(int i = 0; i < 20; i++){
        vec3 dir = cosineweighted(norm.xzy,r).xzy;
        vec3 currpos = pos + dir*rndf(r)*1.3;
        vec4 newcoords = gbufferProjection*gbufferModelView*vec4(currpos.xyz, 1.);
        newcoords /= newcoords.w;
        newcoords.xyz = newcoords.xyz * 0.5 + 0.5;
        float Depth1 =  texture2D(colortex2, newcoords.xy*0.5).w;
        if(Depth1 < newcoords.z && abs(Depth1 - newcoords.z) < 0.01){
            ambientO += 1.0*(1.0-Depth*0.3);
        }

    }
    ambientO/=20.;



    data0 = normalsAndDepth;

      //  gl_FragData[3] = texture2D(colortex3, TexCoords);


    float variance = 0.;
    vec3 KeepColor = vec3(0.);
    //if(CheckIfSky != 20.5){
        if(ProjectedCoordinates.x > 1.0 || ProjectedCoordinates.x < 0.0 || ProjectedCoordinates.y > 1.0 || ProjectedCoordinates.y < 0.0 ){
            data7 =  vec4(texture2D(colortex1, TexCoords*0.5).xyz, 1.0);
            KeepColor = texture2D(colortex1, TexCoords*0.5).xyz;
            //gl_FragData[3] = vec4(texture2D(colortex3, TexCoords).xyz, 0.0f);
            variance = 1.;
            finalWeigth = 0.0;
                

        }else{
            float mm = clamp(MN(clamp(length(ProjectedCoordinates-TexCoords)*13.3,0., 2.), 0.05, 0.2), 0.2, 0.99);
            KeepColor = texture2D(colortex1, TexCoords*0.5).xyz*(0.05) + texture2D(colortex7, ProjectedCoordinates.xy).xyz*0.95;

           // KeepColor = mix(texture2D(colortex1, TexCoords).xyz,texture2D(colortex7, ProjectedCoordinates.xy).xyz,finalWeigth);
           float avgLum = lum(texture2D(colortex1, TexCoords*0.5).xyz)*0.07 + texture2D(colortex7, ProjectedCoordinates.xy).w*0.93;

           float amb = clamp(min(texture2D(colortex10, TexCoords*0.5).w, texture2D(colortex7, ProjectedCoordinates.xy).w), 0.3, 1.0);

            data7 = vec4(KeepColor, amb);
            
          // variance = pow(lum(texture2D(colortex1, TexCoords*0.5).xyz) - avgLum,2.)*0.07 + texture2D(colortex4, ProjectedCoordinates.xy).w*0.93;
           // gl_FragData[3] = vec4(texture2D(colortex3, TexCoords).xyz, variance);

            ambientO = ambientO*0.1 + texture2D(colortex11, ProjectedCoordinates.xy).w*0.9;

        }
    //}




    uvec2 data = uvec2(texelFetch(shadowcolor0, ivec2(TexCoords*2048), 0).xy);
    //vec2 data = texture2D(shadowcolor0, TexCoords).xy;
    float dist = float(data.y);
    data2 = texture2D(colortex2, TexCoords);
    data11 = vec4(ambientO);
    data4 = vec4(KeepColor, texture2D(colortex4, TexCoords*0.5).w);
}