#version 450 compatibility
#extension GL_ARB_shading_language_packing: enable



	/* DRAWBUFFER : 012347 */


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

uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex12;

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
void main(){
    // Account for gamma correction
    vec3 eyeCam = cameraPosition + gbufferModelViewInverse[3].xyz;
    vec4 normalsAndDepth = texture2D(colortex0, TexCoords*0.5);
float CheckIfSky =  texture2D(colortex5, TexCoords*0.5).w;
float Depth =  texture2D(colortex2, TexCoords*0.5).w;


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

    gl_FragData[0] = normalsAndDepth;

      //  gl_FragData[3] = texture2D(colortex3, TexCoords);


    float variance = 0.;
    vec3 KeepColor = vec3(0.);
    if(CheckIfSky != 20.5){
        if(ProjectedCoordinates.x > 1.0 || ProjectedCoordinates.x < 0.0 || ProjectedCoordinates.y > 1.0 || ProjectedCoordinates.y < 0.0){
            gl_FragData[7] =  vec4(texture2D(colortex1, TexCoords*0.5).xyz, lum(texture2D(colortex1, TexCoords*0.5).xyz));
            KeepColor = texture2D(colortex1, TexCoords*0.5).xyz;
            //gl_FragData[3] = vec4(texture2D(colortex3, TexCoords).xyz, 0.0f);
            variance = 1.;
            finalWeigth = 0.0;
        }else{
            KeepColor = texture2D(colortex1, TexCoords*0.5).xyz*0.07 + texture2D(colortex7, ProjectedCoordinates.xy).xyz*0.93;
           // KeepColor = mix(texture2D(colortex1, TexCoords).xyz,texture2D(colortex7, ProjectedCoordinates.xy).xyz,finalWeigth);
           float avgLum = lum(texture2D(colortex1, TexCoords*0.5).xyz)*0.07 + texture2D(colortex7, ProjectedCoordinates.xy).w*0.93;
            gl_FragData[7] = vec4(KeepColor, avgLum);
            
          // variance = pow(lum(texture2D(colortex1, TexCoords*0.5).xyz) - avgLum,2.)*0.07 + texture2D(colortex4, ProjectedCoordinates.xy).w*0.93;
           // gl_FragData[3] = vec4(texture2D(colortex3, TexCoords).xyz, variance);

        }
    }
    //gl_FragData[7] = vec4(Depth);

    uvec2 data = uvec2(texelFetch(shadowcolor0, ivec2(TexCoords*2048), 0).xy);
    //vec2 data = texture2D(shadowcolor0, TexCoords).xy;
    float dist = float(data.y);
    gl_FragData[2] = texture2D(colortex2, TexCoords);
    //gl_FragData[1] = vec4(unpackUnorm4x8((data.x)).xyz,dist/20.);

    gl_FragData[4] = vec4(KeepColor, variance);
}