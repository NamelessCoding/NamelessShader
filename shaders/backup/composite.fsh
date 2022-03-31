#version 120

#include "distort.glsl"

varying vec2 TexCoords;

// Direction of the sun (not normalized!)
uniform vec3 sunPosition;

// The color textures which we wrote to
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D depthtex0;
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D noisetex;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

/*
const int colortex0Format = RGBA16F;
const int colortex1Format = RGB16;
const int colortex2Format = RGB16;
*/

const float sunPathRotation = -40.0f;
const int shadowMapResolution = 2048;
const int noiseTextureResolution = 64;

const float Ambient = 0.025f;

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

float Visibility(in sampler2D ShadowMap, in vec3 SampleCoords) {
    return step(SampleCoords.z - 0.001f, texture2D(ShadowMap, SampleCoords.xy).r);
}

vec3 TransparentShadow(in vec3 SampleCoords){
    float ShadowVisibility0 = Visibility(shadowtex0, SampleCoords);
    float ShadowVisibility1 = Visibility(shadowtex1, SampleCoords);
    vec4 ShadowColor0 = texture2D(shadowcolor0, SampleCoords.xy);
    vec3 TransmittedColor = ShadowColor0.rgb * (1.0f - ShadowColor0.a); // Perform a blend operation with the sun color
    return mix(TransmittedColor * ShadowVisibility1, vec3(1.0f), ShadowVisibility0);
}

#define SHADOW_SAMPLES 2
const int ShadowSamplesPerSize = 2 * SHADOW_SAMPLES + 1;
const int TotalSamples = ShadowSamplesPerSize * ShadowSamplesPerSize;

vec3 GetShadow(float depth) {
    vec3 ClipSpace = vec3(TexCoords, depth) * 2.0f - 1.0f;
    vec4 ViewW = gbufferProjectionInverse * vec4(ClipSpace, 1.0f);
    vec3 View = ViewW.xyz / ViewW.w;
    vec4 World = gbufferModelViewInverse * vec4(View, 1.0f);
    vec4 ShadowSpace = shadowProjection * shadowModelView * World;
    ShadowSpace.xy = DistortPosition(ShadowSpace.xy);
    vec3 SampleCoords = ShadowSpace.xyz * 0.5f + 0.5f;
    float RandomAngle = texture2D(noisetex, TexCoords * 20.0f).r * 100.0f;
    float cosTheta = cos(RandomAngle);
	float sinTheta = sin(RandomAngle);
    mat2 Rotation =  mat2(cosTheta, -sinTheta, sinTheta, cosTheta) / shadowMapResolution; // We can move our division by the shadow map resolution here for a small speedup
    vec3 ShadowAccum = vec3(0.0f);
    for(int x = -SHADOW_SAMPLES; x <= SHADOW_SAMPLES; x++){
        for(int y = -SHADOW_SAMPLES; y <= SHADOW_SAMPLES; y++){
            vec2 Offset = Rotation * vec2(x, y);
            vec3 CurrentSampleCoordinate = vec3(SampleCoords.xy + Offset, SampleCoords.z);
            ShadowAccum += TransparentShadow(CurrentSampleCoordinate);
        }
    }
    ShadowAccum /= TotalSamples;
    return ShadowAccum;
}



float intersect(vec3 p, vec3 d, vec4 sphere){
vec3 cam = p;
for(int i = 0; i < 280; i++){
float dist = -(length(sphere.xyz-p)-sphere.w);
if(dist < 0.01){
return length(p-cam);
}
p += d*dist;
}
return 0.;
}

float RayleighPhase(float cost){
return (3./(16.*3.14159))*(1.0+cost*cost);
}

float p22(float h){
return exp(-h/8500.);
}

vec3 S(vec3 wave, float cost, float h){
float n = 1.00029;
float N = 2.504;
float a = 3.14159*3.14159*(n*n-1.0)*(n*n-1.0)/2.0;
float b = p22(h)/N;
vec3 c = 1.0/(wave*wave*wave*wave);
float d = (1.0-cost*cost);
return a*b*c*d;
}

vec3 B(vec3 wave, float h){
float n = 1.00029;
float N = 2.504;

float a = 8.*pow(3.14159,3.)*(n*n-1.0)*(n*n-1.0)/3.;
float b = p22(h)/N;
vec3 c = 1./wave;
return a*b*c;
}

vec3 reyleighapprox(vec3 wave, float cost, float h){
float n = 1.00029;
float d = 1432.1;
float a = (1.0+cost*cost)/(2.0*h*h);
vec3 b = pow(2.*3.14159/wave,vec3(4.));
float c = pow((n*n-1.0)/(n*n+2.),2.);
float d2 = pow(d/2.,6.);
return a*b*c*d2;
}

float CornetteShanksPhase(float costheta, float g){
float a = 3./(8.*3.14159);
float b = (1.0-g*g)*(1.0+costheta*costheta);
float c = (2.0+g*g)*pow(1.0+g*g-2.*g*costheta, 3./2.);
return a*(b/c);
}

float dr(float h){
return exp(h/1200.);
}

float dr2(float h){
return exp(h/8000.);
}

vec3 boreyleigh(float costheta, vec3 wave){
float n = 1.00029;
float N = 1.504;
float a = 1.0+costheta*costheta;
float v = 3.14159*3.14159*pow(n*n-1., 2.);
return (v/(3.*N*wave*wave*wave*wave*0.000000000002))*a;
}

vec3 boreyleighconstant(vec3 wave){
float n = 1.00029;
float N = 1.504;
float v = 8.*pow(3.14159, 3.)*pow(n*n-1., 2.);
return (v/(3.*N*wave*wave*wave*wave));
}

vec3 bommie(float costheta, vec3 wave){
float T = 5.;
float C = (0.6544*T-0.6510);
vec3 Bm = 0.434*C*3.14159*((4.*3.14159*3.14159)/(wave*wave))*0.67;
return 0.434*C*((4.*3.14159*3.14159)/(wave*wave))*0.5*Bm;
}
vec3 bommieconstant(vec3 wave){
float T = 5.;
float C = (0.6544*T-0.6510);
vec3 Bm = 0.434*C*3.14159*((4.*3.14159*3.14159)/(wave*wave))*0.67;
return Bm;
}

float HenyeyG(float cost, float g){
return (1.0-g*g)/(4.*3.14159*pow(1.0+g*g-2.*cost,3./2.));
}

float ph(float h, float H){
return exp(-(abs(h)/H));
}

vec3 S(vec3 wave, float h, float H, float cost){
float n = 1.00029;
float N = 2.504*pow(10., 2.);
float a = pow(3.14159,2.)*pow(n*n-1.,2.)/2.;
float b = ph(h, H)/N;
vec3 c = 1./pow(wave, vec3(4.));
float d = (1.0+cost*cost);
return a*b*c*d;
}
//S = B*Y

vec3 B(vec3 wave, float h, float H){
float n = 1.00029;
float N = 2.504*pow(10., 2.);
float a = 8.*pow(3.14159,3.)*pow(n*n-1.,2.)/3.;
float b = ph(h, H)/N;
vec3 c = 1./pow(wave, vec3(4.));
return a*b*c;
}

vec3 Bconstant(vec3 wave){
float n = 1.00029;
float N = 2.504*pow(10., 25.);
float a = 8.*pow(3.14159,3.)*pow(n*n-1.,2.)/3.;
float b = 1./N;
vec3 c = 1./pow(wave, vec3(4.));
return a*b*c;
}

vec3 BsR(vec3 wave, float h, float H){
float n = 1.00029;
vec3 N = vec3(2.504)*pow(10., 25.);
vec3 a = (8.*pow(3.14159,3.)*pow(n*n-1.,2.))/(3.*N*pow(wave,vec3(4.)));
float b = exp(-(h/H));
return a*b;
}

float PM(float cost, float g){
float a = 3./(8.*3.14159);
float b = (1.0-g*g)*(1.0+cost*cost);
float c = (2.0+g*g)*pow(1.0+g*g-2.*g*cost, 3./2.);
return a*(b/c);
}

float PR(float cost){
return (3./(16.*3.14159))*(1.0+cost*cost);
}

float Y(float cost){
float a = 3./(16.*3.14159);
float b = 1.0+cost*cost;
return a*b;
}

vec3 F(vec3 wave,vec3 wave2, vec3 wave3, vec3 wave4, float s, float cost){
vec3 br = boreyleighconstant(wave);
vec3 bm = bommieconstant(wave2);

float pr = ph(s, 8500.);
float pm = ph(s, 1200.);

vec3 Br = boreyleigh(cost, wave3);
vec3 Bm = bommie(cost, wave4);

return pr*Br*br*200. + pm*Bm*bm;
}

vec3 sky(vec3 p, vec3 d, vec3 lig){

float l = intersect(p, d, vec4(0., 0.,0.,6420.));
vec3 div = ((p+d*l)-p)/40.;
vec3 wavelengths = vec3(680., 590., 420.);
float accum = 0.;
vec3 energy = vec3(0.);
vec3 energy2 = vec3(0.);
vec3 br = boreyleighconstant(wavelengths.zyx*0.0007);
vec3 bm = bommieconstant(wavelengths*0.024);
vec3 waves = vec3(0.00000419673, 0.0000121427, 0.0000296453);
float accum3 = 0.;
vec3 m = p;
for(int i = 0; i < 40; i++){
accum += exp(-(p.z-2500.)/8500.)*length(div);
accum3 += exp(-(p.z-2500.)/6500.)*length(div);

float accum2 = 0.;
float accum4 = 0.;

vec3 cam = p;

float l2 = intersect(p, lig, vec4(0.,0.,0.,6420.));
vec3 div2 = ((cam+lig*l2)-cam);
float CP = length(div2);


div2 /= 40.;
if(p.z-6200. < 0.05){
break;
}
if(l2 > 0. ){
//accum2 += exp(-length());
    for(int k = 0; k < 40; k++){
        accum2 += exp(-(max(cam.z-2500., 0.)/6420.)*3.)*length(div2);
        accum4 += exp(-(cam.z-6500.)/6500.)*length(div2);

        cam += div2;
    }
   // accum2 /= 40.;
    //energy += S(wavelengths, max(dot(d,lig),0.), length(p)-6500.);
    
    //energy += reyleighapprox(wavelengths*0.0000004, max(dot(d,lig),0.), length(p));
    energy +=
    exp(-waves*111112.1*accum2)*exp(-waves*11.1*accum)*length(div)*
    reyleighapprox(wavelengths*.0067, max(dot(d,lig),0.001), exp(-(length(p-m))/6110.))
    *RayleighPhase(max(dot(d,lig),0.01))*waves*0.0001;
    
    energy2 += exp(-bm*accum2*1.168)*exp(-bm*accum*4.168)
    *length(div)*bm*1.*PM(max(dot(d,lig),0.), 0.76);
    
   // energy += S(wavelengths*0.0004127, max(dot(d,lig),0.1), exp(-(length(p-m)-1500.)/5500.))
    //*exp(-waves*.01*accum)*length(div)*21.5;
//}
}

    //vec3 S(vec3 wave, float cost, float h){

    
//}
p+=div;
}
return (energy*.0136 +energy2);
}


float remap(float v, float l0, float h0, float ln, float hn){
return ln + ((v-l0)*(hn-ln))/(h0-l0);
}
vec3 remap(vec3 v, vec3 l0, vec3 h0, vec3 ln, vec3 hn){
return ln + ((v-l0)*(hn-ln))/(h0-l0);
}
float random3d(vec3 p){
return fract(sin(p.x*214. + p.y*241. + p.z*123.)*100. + cos(p.x*42. + p.y*41.2+p.z*32.)*10.);
}

float worley3d(vec3 p){
vec3 f = floor(p);

float ll = 999.;
for(int i = 0; i < 27; i++){
vec3 coords = vec3(float(i%3)-1., mod(float(i/3)-1., 3.), float(i/9)-1.);
vec3 col = f+coords;
vec3 curr = vec3(random3d(col), random3d(col+2.), random3d(col+4.))-0.5;
float len = length((col+curr)-p);
ll = min(ll, len);
}
return ll;

}
float hash(vec3 p3)
{
    p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float noise222( in vec3 x )
{
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    return mix(mix(mix( hash(i+vec3(0,0,0)), 
                        hash(i+vec3(1,0,0)),f.x),
                   mix( hash(i+vec3(0,1,0)), 
                        hash(i+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(i+vec3(0,0,1)), 
                        hash(i+vec3(1,0,1)),f.x),
                   mix( hash(i+vec3(0,1,1)), 
                        hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}

float fbmss(vec3 p)
{
    float scale = 0.1;
    float threshold = 0.3;
    float sum = 0.;
        p += vec3(5., 0., 0.);
        for(int i = 1; i <= 8; i++)
        {
            sum += noise222(p * scale * pow(2., float(i))) / pow(1.7, float(i));
        }
        return max(sum - threshold, 0.);
}
float fbm(vec3 p, vec3 cam){
//p.yz = rot(p.yz, iTime*0.3);
//float a = texture(iChannel0, p).x*0.5 +texture(iChannel0, p*2.).y*0.25+
//texture(iChannel0, p*4.).z*0.125+texture(iChannel0, p*8.).x*0.0625;
//float a = texture(iChannel0, p).x*0.9;

float b = fbmss(p*232.);

float Srb = clamp(remap((cam.z-6500.)/2000., 0., 0.07, 0., 1.),0.,1.);
//a *= clamp(abs(length(p)-6500.)*0.00013, 0.0, 1.);
//a -= clamp((p.z)*0.4,0., 1.);
//a = max(a,0.);
//vec3 pos = vec3(0.,0.,6500.)-p;
//pos.xz = rot(pos.xz, iTime*200.);
//float cap = box(pos, vec3(100.,500.,100.));
//cap = capsule(pos, vec3(0.,-2000., 6500.), vec3(0.,2000.,6500.), 100.);
//cap = abs(cap)+0.01;
//float density = exp(-cap*0.00002);
return clamp(((Srb)*b*clamp((cam.z-6500.)*0.44,0.,1.)),0.,1.)+clamp(sin(p.x)+sin(p.y)+sin(p.z),0.,1.);
}

float noise(vec3 p){
return fract(sin(p.x * 23. + p.y * 241.4 + p.z*52.)*100. + cos(p.x * 234. + p.y * 21.4 + p.z*542.)*124.);
}

//float PM(float cost, float g){
//float a = 3./(8.*3.14159);
//float b = ((1.0-g*g)*(1.0+cost*cost))/((2.0+g*g)*pow(1.0+g*g-2.*g*cost, 3./2.));
//return a*b;
//}
//my own code as well, just can't find the math
void swap(inout float t1, inout float t2) {
  float m = t1;
  t1 = t2;
  t2 = m;
}
bool intersectB(vec3 RayPosition, vec3 rayDir, inout float tmin, inout float tmax)
// bool intersect(const Ray &r)
{
  vec3 orig = RayPosition;
  vec3 dir = rayDir;
  tmin = (-15000. - orig.x) / dir.x;
  tmax = (15000. - orig.x) / dir.x;

  if (tmin > tmax)
    swap(tmin, tmax);

  float tymin = (-15000. - orig.y) / dir.y;
  float tymax = (15000. - orig.y) / dir.y;

  if (tymin > tymax)
    swap(tymin, tymax);

  if ((tmin > tymax) || (tymin > tmax))
    return false;

  if (tymin > tmin)
    tmin = tymin;

  if (tymax < tmax)
    tmax = tymax;

  float tzmin = (6500. - orig.z) / dir.z;
  float tzmax = (6400. - orig.z) / dir.z;

  if (tzmin > tzmax)
    swap(tzmin, tzmax);

  if ((tmin > tzmax) || (tzmin > tmax))
    return false;

  if (tzmin > tmin)
    tmin = tzmin;

  if (tzmax < tmax)
    tmax = tzmax;

  return true;
}

vec4 clouds2(vec3 p, vec3 d, vec3 lig, inout float dist){
vec3 waves = vec3(0.00000519673, 0.0000121427, 0.0000296453);
float ttt = 0.;
  float tttm = 0.;
  bool isHit = intersectB(p, d, ttt, tttm);
  if (!isHit) {
    return vec4(0., 0., 0., 1.);
 } else {
    p += d * ttt;
  }
float transmission = 1.0;
vec3 Ex = vec3(1.0);

float phase = PM(max(dot(d,lig),0.), 0.76);

vec3 wavelengths = vec3(680., 550., 440.);

vec2 t = vec2(0.);
vec3 energy = vec3(1.);//0000296453
vec3 rayleighcoefficients = vec3(0.00000519673, 0.0000121427, 0.0000296453);
vec3 T = vec3(0.);
float reyleighH = 8500.;
float MieH = 1200.;

vec3 accumulateLight = vec3(0.);
vec3 accumulateLightMie = vec3(0.);
vec3 accumother = vec3(0.);
//if(intersect(p, vec3(0.,0.,0.), 8500.0, d, t)){
//col = vec3(t.x);
vec3 ccc = p;
vec3 cam = p;
vec3 fin = p+d*t.y;
vec3 div = vec3(fin-cam) / 40.;
//vec3 precomputed = ((vec3(5.8,13.5,33.1)))*exp(-6.);
//vec3 precomputed2 = vec3(0.210)*exp(-5.);
float mm = length(cam-fin);
//vec3 energyLoss = exp(-rayleighcoefficients*mm);
float Is = 3.;
vec3 Ip = vec3(0.);
vec3 accum = vec3(0.);
float accum11 = 0.;
float minus = 0.42;
float mult = 0.00001;

float zz = max(dot(vec3(0.,0.,1.),lig),0.);
/////////////
//vec3 br = boreyleighconstant(wavelengths.zyx*0.0005);
vec3 bm = bommieconstant(wavelengths*0.024);

////////////
//float pm = PM(max(dot(vec3(0.,0.,1.),lig),0.), 0.76)*5.;
//float pr = PR(max(dot(vec3(0.,0.,1.),lig),0.))*2.;
float pm2 = PM(max(dot(d,lig),0.), 0.76)*7.;
vec3 sky2 = sky(vec3(0., 0., 6380.), normalize(vec3(0.,1.0,0.0)), lig);

vec3 pr2 = PR(max(dot(d,lig),0.))*vec3(4.);
float keepdensity = 0.;
bool firsth = false;
for(int i = 0; i < 70; i++){
//accum += ph(length(cam), reyleighH)*length(div);
float density = max(fbm(cam*mult, cam)-minus-abs(cam.z-6500.)*0.00034, 0. );
//density = smoothstep(0.,1.,density);
density = clamp(density, 0., 1.);
keepdensity += density;
if(density > 0.01){
if(!firsth){
dist = length(cam-ccc);
firsth = true;
}
accum += density*50.6;
//accum11 += ph(length(cam)-6500., MieH)*length(div);

vec3 accum2 = vec3(0.);
//float accum3 = 0.;
//energy = energy*(1.0-rayleighcoefficients);
        vec3 cam2 = cam;
        
        for(int k = 0; k < 10; k++){
            float density2 = max(fbm(cam2*mult, cam2)-minus-abs(cam2.z-6500.)*0.00034,0.) ;
            //density2 = smoothstep(0.,1.,density2);
density2 = clamp(density2, 0., 1.);

            //accum2 += ph(cam2.z, 1300.)*30.;
            accum2 += density2*30.1;
            //accum3 += ph(length(cam2)-6500., MieH)*length(div2);
            cam2 += lig*30.1;

        }



Ex = Ex*exp(-accum*0.01)*(1.0-exp(-accum*300.));
transmission*= 1.0-density;
transmission *= 0.99;
if(transmission < 0.1){
break;
}

 // sky(vec3 p, vec3 d, vec3 lig){
accumulateLight += density * max(Ex, 0.03) *
                        (pm2 * exp(-.4 * accum2) * 11.4 * vec3(0.9, 0.6, 0.2) + pr2 * exp(-.01  * accum2) + exp(-.6 * accum2 * (1.0 - zz * 0.9)) +
                          exp(-accum2 * 5.1 ) * mix(bm * 23., vec3(12.), smoothstep(0., 1., zz))) *
                         1.8 ;

    
   }

    cam += d * (50.6 - 10. * random3d(cam));
    // if(length(cam)>7500.){break;}
  }
 // accumulateLight *= sky2;
  return vec4(accumulateLight * 1.3 
 , transmission*Ex);
}

vec3 F(vec3 A, vec3 B, vec3 C, vec3 D, vec3 E, float cost, float y){
return (1.0+A*exp(B/cost))*(1.0+C*exp(D*y) + E*cos(y)*cos(y));
}
vec3 skyp3(vec3 d, vec3 lig){
float cost = max(dot(lig, vec3(0.,0.,1.)),0.);
float thetaS = acos(cost);
float cosp = max(dot(d, vec3(0.,0.,1.)),0.);
float thetaP = acos(cosp);
float y = acos(max(dot(d,lig),0.));
float ycos = cos(y);

float T = 2.;

float Yz = (4.0453*T-4.9710)*tan((4./9. - T/120.)*(3.14159-2.*thetaS))-0.2155*T+2.4192;

float tt = thetaS*thetaS;
float ttt =  thetaS*thetaS*thetaS;
vec3 xA = vec3(0.00166*ttt-0.00375*tt+0.00209*thetaS,
-0.02903*ttt + 0.06377*tt - 0.03202*thetaS + 0.00394,
0.11693*ttt - 0.21196*tt + 0.06052*thetaS + 0.25886
);
float xz = xA.x*T*T + xA.y*T + xA.z;

vec3 yA = vec3(0.00275*ttt-0.00610*tt+0.00317*thetaS,
-0.04214*ttt + 0.08970*tt - 0.04153*thetaS + 0.00516,
0.15346*ttt - 0.26756*tt + 0.06670*thetaS + 0.26688
);
float yz = yA.x*T*T + yA.y*T + yA.z;

vec3 A = vec3(0.1787*T-1.4630, -0.0193*T-0.2592, -0.0167*T-0.2608);
vec3 B = vec3(-0.3554*T+0.4275, -0.0665*T+0.0008, -0.0950*T+0.0092);
vec3 C = vec3(-0.0227*T+5.3251, -0.0004*T+0.2125, -0.0079*T+0.2102);
vec3 D = vec3(0.1206*T-2.5771, -0.0641*T-0.8989, -0.0441*T-1.6537);
vec3 E = vec3(-0.0670*T+0.3703, -0.0033*T+0.0452, -0.0109*T+0.0529);

vec3 Ff = F(A,B,C,D,E,cosp, y);
vec3 Ff2 = F(A,B,C,D,E,cos(0.), thetaS);
//return Ff2;
float Y = (Yz*Ff.x)/Ff2.x;
float x = (xz*Ff.y)/Ff2.y;
float ys = (yz*Ff.z)/Ff2.z;

vec3 XYZ = vec3((x*Y)/ys, Y, (((1.0-x-ys)*Y)/ys)).xyz;
vec3 RGB = vec3(3.2404542*XYZ.x - 1.5371385*XYZ.y - 0.4985314*XYZ.z,
-0.9692660*XYZ.x + 1.8760108*XYZ.y + 0.0415560*XYZ.z,
0.0556434*XYZ.x - 0.2040259*XYZ.y + 1.0572252*XYZ.z
);
vec3 final = RGB.xyz*0.03;
float dist = 0.;
vec4 mmm = clouds2(vec3(0.,200.,5500.),d,lig,dist);
vec3 skys = final;
final = final*mmm.w + mmm.xyz*0.9;
float f = exp(-dist * 0.00048);
final.xyz = final.xyz * f + skys.xyz * (1.0 - f);
//float zz = max(dot(lig, vec3(0., 0., 1.)), 0.);
//final.xyz *= clamp(zz * 3., 0.2, 1.);
//final*=0.9;
///vec4 c = clouds2(vec3(0.,0.,5200.),d,lig);
//final = final*c.w + c.xyz;
return final;
}

//NOT MY CODE//////////////////////
vec3 ACESFilm(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.,1.);
}
float linearizeDepthFast(float depth, float near, float far) {

    return (near * far) / (depth * (near - far) + far);

}
//////////////////////////////////


void main(){
    // Account for gamma correction
    vec3 Albedo = pow(texture2D(colortex0, TexCoords).rgb, vec3(2.2f));
    float Depth = texture2D(depthtex0, TexCoords).r;


vec3 sunPos = mat3(gbufferModelViewInverse) * normalize(sunPosition);

    if(Depth == 1.0f){
       vec3 screenPos = vec3(TexCoords, texture2D(depthtex0, TexCoords).r);
        vec3 clipPos = screenPos * 2.0 - 1.0;
        vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
        vec3 viewPos = tmp.xyz / tmp.w;
        viewPos = mat3(gbufferModelViewInverse) * viewPos;


        vec3 sky2 = skyp3(normalize(viewPos.xzy),normalize(sunPos).xzy);

        //vec3 sky2 = skyp3(normalize(pos.xzy), normalize(vec3(0., 0.6, 0.1)));
    	sky2 = pow(sky2, vec3(1.6));
        sky2 = ACESFilm(sky2);
        sky2 = pow(sky2, vec3(1./2.2));

        gl_FragData[0] = vec4(sky2*0.9, 1.0f);
        return;
    }
    vec3 sky2 = skyp3(normalize(vec3(0.,0.6,0.1)), normalize(sunPos).xzy);
    // Get the normal
    vec3 Normal = normalize(texture2D(colortex1, TexCoords).rgb * 2.0f - 1.0f);
    // Get the lightmap
    vec2 Lightmap = texture2D(colortex2, TexCoords).rg;
    vec3 LightmapColor = GetLightmapColor(Lightmap);
    // Compute cos theta between the normal and sun directions
    float NdotL = max(dot(Normal, normalize(sunPosition)), 0.0f);
    // Do the lighting calculations
    vec3 Diffuse = Albedo * (LightmapColor + NdotL * GetShadow(Depth)*2.0f + Ambient*6.0);
    /* DRAWBUFFERS:0 */
    // Finally write the diffuse color

float dist = linearizeDepthFast(Depth, 2.0f, 100.0f);

dist *= 0.029;
float f = exp(-dist*dist*dist*0.01);
Diffuse = Diffuse*f + sky2*(1.0-f)*2.0;
    //Diffuse = pow(Diffuse, vec3(1.5));

    
    gl_FragData[0] = vec4(Diffuse, 1.0f);
}