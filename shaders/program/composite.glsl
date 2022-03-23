
/*
const int shadowcolor1Format = RGBA32F;
const int shadowcolor0Format = RG32UI;
const float sunPathRotation = -45.0;
const bool shadowtex0Clear = true;
const int shadowtex0Format = RGBA32F;
const vec4 shadowcolor0ClearColor = vec4(0,0,0,0);
const int colortex1Format = RGBA32F;
const bool colortex1Clear = false;
const int colortex5Format = RGBA32F;
const bool colortex5Clear = false;
const int colortex4Format = RGBA32F;
const bool colortex4Clear = false;
const int colortex7Format = RGBA32F;
const bool colortex7Clear = false;
const int colortex3Format = RGBA32F;
const bool colortex3Clear = true;
const int colortex0Format = RGBA32F;
const bool colortex0Clear = false;
const int colortex6Format = RGBA32F;
const bool colortex6Clear = true;
const int colortex2Format = RGBA32F;
const bool colortex2Clear = false;
*/


#if STAGE == STAGE_VERTEX

	in vec3 at_midBlock;

	out vec2 texcoord;
	//out vec3 midBlock;

	void main() {
		gl_Position = ftransform();
	    texcoord = gl_MultiTexCoord0.st;
	}

#elif STAGE == STAGE_FRAGMENT

	in vec2 texcoord;
	//in vec3 midBlock;

	
	/* DRAWBUFFER : 012356 */

	uniform sampler2D colortex1;
	uniform sampler2D colortex4;
    uniform sampler2D colortex3;
    uniform sampler2D colortex2;
    uniform sampler2D colortex0;

	uniform sampler2D depthtex0;
	uniform sampler2D depthtex1;
	uniform sampler2D depthtex2;
	uniform usampler2D shadowcolor0;
	uniform usampler2D shadowcolor1;
	uniform sampler2D shadowtex0; // Needed to enable shadow maps
    uniform float frameTime;
    uniform vec2 viewResolution;
uniform vec3 sunPosition;
uniform int frameCounter;

	uniform vec3 cameraPosition;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;


uniform mat4 gbufferProjection;

uniform mat4 shadowProjection;
uniform mat4 shadowModelViewInverse;
uniform vec3 previousCameraPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform sampler2D shadowtex1; // Needed to enable shadow maps


uniform mat4 shadowModelView;

uniform float near;
uniform float far;
	#include "/lib/voxelization.glsl"
	#include "/lib/raytracer.glsl"

	struct Material {
		vec3   albedo;       // Scattering albedo
		float  opacity;
		vec3   normal;       // Normal
		float spec;
	};

	Material getMaterial(vec4 albedo, vec4 specular, vec4 normal, int voxelID) {
		Material material;
		
		material.albedo      = albedo.rgb;
		material.opacity     = albedo.a;
	material.spec = specular.x;
		material.normal      = normal.rgb;

		return material;
	}

	Material getMaterial(vec3 position, vec3 normal, vec4[2] voxel) {
		int id = ExtractVoxelId(voxel);

		// Figure out texture coordinates
		int   tileSize = ExtractVoxelTileSize(voxel);
		ivec2 tileOffs = ExtractVoxelTileIndex(voxel) * tileSize;

		ivec2 tileTexel;
		mat3 tbn;

		if (abs(normal.y) > abs(normal.x) && abs(normal.y) > abs(normal.z)) {
			tileTexel = ivec2(fract(position.x) * tileSize, fract(position.z * sign(normal.y)) * tileSize);
			tbn = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, sign(normal.y)), normal);
		} else {
			tileTexel = ivec2(fract(position.x * sign(normal.z) - position.z * sign(normal.x)) * tileSize, fract(-position.y) * tileSize);
			tbn = mat3(vec3(sign(normal.z), 0.0, -sign(normal.x)), vec3(0.0, -1.0, 0.0), normal);
		}

		ivec2 texelIndex = tileOffs + tileTexel;

		// Read textures
		vec4 baseTex     = texelFetch(depthtex0, texelIndex, 0);
		vec4 specularTex = texelFetch(depthtex1, texelIndex, 0);
		vec4 normalTex   = texelFetch(depthtex2, texelIndex, 0);

		normalTex.xyz = normalTex.xyz * 2.0 - 1.0;
		normalTex.xyz = normalize(tbn * normalTex.xyz);

		baseTex.rgb *= ExtractVoxelTint(voxel);

		return getMaterial(baseTex, specularTex, normalTex, id);
	}


bool IntersectVoxelAlphatest(vec3 origin, ivec3 index, vec3 direction, vec4[2] voxel, int id, out vec3 hitPos, inout vec3 hitNormal) {
 	if (IntersectVoxel(origin, index, direction, id, hitPos, hitNormal)) {
 		// Perform alpha test
 		Material voxelMaterial = getMaterial(hitPos, hitNormal, voxel);
 		return (voxelMaterial.opacity > 0.102);
 	}

 	return false;
 }

bool RaytraceVoxel(vec3 origin, ivec3 originIndex, vec3 direction, bool transmit, const int maxSteps, out vec4[2] voxel, out vec3 hitPos, out ivec3 hitIndex, out vec3 hitNormal, out int id) {
	
	voxel = ReadVoxel(originIndex);
	hitIndex = originIndex;

    id = ExtractVoxelId(voxel);
	if (id > 0 && !transmit ) {
		//bool IntersectVoxelAlphatest(vec3 origin, ivec3 index, vec3 direction, int id, out vec3 hitPos, inout vec3 hitNormal) {

		if (IntersectVoxelAlphatest(origin, originIndex, direction,voxel, id, hitPos, hitNormal)) {
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

		if (!IsInVoxelizationVolume(hitIndex)) { break; }

		voxel = ReadVoxel(hitIndex);
		id = ExtractVoxelId(voxel);
		if (id > 0 ) {
			hit = IntersectVoxelAlphatest(origin, hitIndex, direction,voxel,id, hitPos, hitNormal);
		}
	}

	return hit;
}

	bool rayTrace(vec3 dir,vec3 origin,out vec3 hitPos, out vec3 hitNormal, out vec4[2] voxel, out int id, int stepssize) {
		
		//ec3 startVoxelPosition = SceneSpaceToVoxelSpace(gbufferModelViewInverse[3].xyz);

		ivec3 hitIdx;
		bool hit = RaytraceVoxel(origin, ivec3(floor(origin)), dir, true, stepssize, voxel, hitPos, hitIdx, hitNormal, id);
		return hit;
	}
/***********************************************************************/



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

return final;
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
float minus = 0.52;
float mult = 0.000016;

float zz = max(dot(vec3(0.,0.,1.),lig),0.);
/////////////
//vec3 br = boreyleighconstant(wavelengths.zyx*0.0005);
vec3 bm = bommieconstant(wavelengths*0.024);

////////////
//float pm = PM(max(dot(vec3(0.,0.,1.),lig),0.), 0.76)*5.;
//float pr = PR(max(dot(vec3(0.,0.,1.),lig),0.))*2.;
float pm2 = PM(max(dot(d,lig),0.), 0.76)*7.;
vec3 sky2 = skyp3( normalize(vec3(0.,1.0,0.8)), lig);

vec3 pr2 = PR(max(dot(d,lig),0.))*12.*sky2;
float keepdensity = 0.;
bool firsth = false;
for(int i = 0; i < 170; i++){
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



Ex = Ex*exp(-accum*0.01)*(1.0-exp(-accum*310.8));
transmission*= 1.0-density;
transmission *= 0.99;
if(transmission < 0.1){
break;
}

 // sky(vec3 p, vec3 d, vec3 lig){
accumulateLight += density * Ex *
                        (pm2 * exp(-.4 * accum2) *(11.4+(1.0-zz)*50.)* vec3(0.9, 0.6, 0.2) + pr2 * exp(-.01  * accum2)*3. + exp(-.6 * accum2 * (1.0 - zz * 0.9)) +
                          exp(-accum2 * 5.1 ) * mix(bm * 23., vec3(12.), smoothstep(0., 1., zz))) *
                         1.8 ;

    
   }

    cam += d * (50.6 - 10. * random3d(cam));
    // if(length(cam)>7500.){break;}
  }
 // accumulateLight *= sky2;
  return vec4(accumulateLight * 1.3 *
  mix(reyleighapprox(wavelengths.zyx * 12.9, max(dot(d, lig), 0.), cam.z) * 2., vec3(1.), zz) 
  + accumulateLight*0.5
 , transmission*Ex);
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
float linearizeDepthFast(float depth) {

    return (near * far) / (depth * (near - far) + far);

}
//////////////////////////////////
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


vec3 ggx_S(vec3 d, inout uint r, float a){
float r1 = rndf(r);
float r2 = rndf(r);

float phi = r1*3.14159*2.;
float theta = atan(a*sqrt(r2/(1.0-r2)));

float x = cos(phi)*sin(theta);
float y = sin(phi)*sin(theta);
float z = cos(theta);

vec3 N = d;
vec3 W = (abs(N.x) > 0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
vec3 T = normalize(cross(N,W));
vec3 B = normalize(cross(N,T));

return normalize(T*x + B*y + z*N);
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

float ggx_D(float cost, float a){
float as = a*a;
float of = 3.14159*pow((a*a-1.)*cost*cost + 1.0,2.);
return as/of;
}

float ggx_pdf(float cost, float a){
float as = a*a*cost;
float of = 3.14159*pow((a*a-1.)*cost*cost + 1.0,2.);
return as/of;
}

float cookTorranceG(vec3 n, vec3 h, vec3 v, vec3 l){
return min(1., min((2.*max(dot(n,h),0.)*max(dot(n,v),0.))/max(dot(v,h),0.001),
(2.*max(dot(n,h),0.)*max(dot(n,l),0.))/max(dot(v,h),0.001)));
}

vec3 Schlick(vec3 F0, float cost){
return F0 + (1.0-F0)*pow(1.0-cost,5.);
}



float lerp(float a, float b, float c){
return a+(b-a)*c;
}
vec3 lerp(vec3 a, vec3 b, float c){
return a+(b-a)*c;
}

float powerheuristics(float pdf1, float pdf2){
return (pdf1*pdf1)/(pdf1*pdf1+pdf2*pdf2);
}

vec3 samplesun(vec3 d, inout uint r, float a){
float r1 = rndf(r);
float r2 = rndf(r);
float z = (1.0-cos(a))*r2+cos(a);

float phi = r1*3.14159*2.;

float x = cos(phi)*sqrt(1.0-z*z);
float y = sin(phi)*sqrt(1.0-z*z);

vec3 N = d;
vec3 W = (abs(N.x) > 0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
vec3 T = normalize(cross(N,W));
vec3 B = normalize(cross(N,T));

return normalize(T*x + B*y + z*N);
}


vec3 quickShadow2(vec3 Position){
    //vec3 Position = texture2D(colortex2, texcoord).xyz;

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

vec2 rot(vec2 a, float b){
float l = length(a);
a = normalize(a);
float ang = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
ang += b;
return l*vec2(cos(ang), sin(ang));
}

vec3 quickShadow(){
    vec3 Position = texture2D(colortex2, texcoord).xyz;
	
   //return Position;
//return Depth;
    ///NOT MY CODE////////////////////////////////////
    vec4 View = vec4(Position,1.);

 vec2 ires = vec2(viewWidth, viewHeight);
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
	vec3 accum = vec3(0.);
    vec2 ProjectedCoordinates = Projected.xy;
for(int i = 0; i < 25; i++){
	vec2 coords = vec2(float(i%5)-2., float(i/5)-2.)*0.1;
	coords = rot(coords, float(i)*((2.*3.14159)/25.));
	vec2 fin = (ProjectedCoordinates.xy*ires + coords)/ires;
    float readDepth = texture2D(shadowtex0, fin).x;
    //return vec3(readDepth)*1.6;
//return linearizeDepthFast(Projected.z)*0.6;
    if(abs((readDepth) - (Projected.z)-0.001) > 0.002){
     //   return vec3(0.0);
    }else{
    accum += vec3(2.0);
	}
}
return accum/25.;
}



vec3 renderpixel(vec3 p, vec3 d, inout uint r, inout float distat, inout vec3 ptp, inout vec3 firstcol, inout float spec, inout vec3 firstnorm, inout vec3 prevcol2, inout float distat2){
const int bounces = 7;
vec3 col = vec3(0.);
vec3 col2 = vec3(0.);
vec3 cam = p;
float l = 0.;
vec3 c = vec3(1.);
vec3 tt = vec3(1.);
vec3 sunPos = mat3(gbufferModelViewInverse) * normalize(sunPosition);

vec3 ldir = normalize(sunPos);

vec3 ldir2 = ldir;
vec4[2] voxel;
vec3 n = vec3(0.);

vec3 camera = (p);
vec3 firstHit = camera;
for(int i = 0; i < bounces; i++){
	vec3 origin = p;
	int ID = 0;
bool hit = rayTrace(d ,origin,p, n, voxel, ID, 258);

c = getMaterial(p, n, voxel).albedo;
n = normalize(getMaterial(p, n, voxel).normal);
float specular22 = getMaterial(p,n,voxel).spec;
if((hit && i > 0) || (length(texture2D(colortex3, texcoord).xyz)>0.05 && i == 0) ){
if(i==0){
	vec3 nn = n;
		n = texture2D(colortex0, texcoord).xyz;

firstHit = (p);

	
		spec = specular22;//(specular22<0.1)?0.1:0.9;
spec = 0.05;
	if(ID == 250){
			spec = 250.;
			c = vec3(.99);
			n = nn;
	}else{
			c = texture2D(colortex3, texcoord).xyz;

	p = SceneSpaceToVoxelSpace(texture2D(colortex2, texcoord).xyz);
	specular22 =  texture2D(colortex3, texcoord).w;
	spec = specular22;
	}
	//c = texture2D(colortex3, texcoord).xyz*max(dot(n, ldir2),0.5);
		prevcol2 = c;
	//c = vec3(0.99);

firstnorm = normalize(n);
distat = length((p)-(cam));

  //  return (near * far) / (depth * (near - far) + far);

float distanish =  length(p-cam);
//distat2 = distanish*(near*far);
ptp = VoxelSpaceToWorldSpace(p);


}
p+=n*0.05;
vec3 wi = d;

//|| ID == 201 || ID == 200


if(ID == 58 || ID == 59 || ID == 60 || ID == 61 || ID == 62 || ID == 63 || ID == 64 || ID == 65 || ID == 66 || ID == 67 || ID == 68 || ID == 69 || ID == 70 || ID == 45
|| ID == 46 || ID == 47 || ID == 48 || ID == 49 || ID == 50 || ID == 51 || ID == 52 || ID == 53 || ID == 54 || ID == 55 || ID == 56 || ID == 300  || ID == 57 ){
	if(i == 0){
	col += tt*c*4.0;
	firstcol = col;
	}else{
col2 += tt*c*4.0;

	}

	break;
}
d = cosineweighted(n.xzy,r).xzy;
vec3 brdf = c/3.14159;
vec3 brdf2 = c/3.14159;

float pdf = max(dot(d,n),0.0)/3.14159;
float hmpdf2 = max(dot(d,n),0.0)/3.14159;

float probability =  clamp(specular22,0.1,0.5);//(specular22 < 0.1)?0.1:0.9;
//probability = 0.1;
//probability = 0.1;
//col = vec3(probability);
//break;
float rough =max(1.0-specular22,0.1);// (max(1.0-specular22, 0.1)<0.1)?0.1:0.9;
rough = 0.1;
rough = rough*rough;
if(ID == 250 ){
probability = 0.95;
rough = 0.01;
}
//probability = 0.05;
if(rndf(r) < probability){
//vec3 ggx_S(vec3 d, inout uint r, float a){
	if(i > 0){
d = ggx_S(reflect(wi,n).xzy, r, rough).xzy;
	}else{
		d = reflect(wi,n);
	}
}
vec3 h = normalize(-wi+d);
//float ggx_D(float cost, float a){
float D = ggx_D(max(dot(reflect(wi,n),d),0.), rough);
//float cookTorranceG(vec3 n, vec3 h, vec3 v, vec3 l){
float G = cookTorranceG(n,h,-wi,d);
//vec3 Schlick(vec3 F0, float cost){
vec3 F = Schlick(vec3(0.3), max(dot(-wi,n),0.));
vec3 specular = (D*G*F)/max(4.*max(dot(d,n),0.6)*max(dot(-wi,n),0.),0.001);

brdf = lerp(brdf, specular, probability);
pdf = lerp(pdf, ggx_pdf(max(dot(reflect(wi,n), d),0.), rough), probability);
hmpdf2 = lerp(hmpdf2, ggx_pdf(max(dot(reflect(wi,n), d),0.), rough), probability);

brdf *= 1.0+(1.0*probability*max(dot(d,n),0.));
pdf = max(pdf, 0.0001);


float theta = 0.1;
if(i > 0){
ldir = (samplesun(ldir2.xzy, r, theta)).xzy;
}
vec3 h2 = normalize(-wi+ldir);
//float ggx_D(float cost, float a){
float D2 = ggx_D(max(dot(reflect(wi,n),ldir),0.), rough);
//float cookTorranceG(vec3 n, vec3 h, vec3 v, vec3 l){
float G2 = cookTorranceG(n,h2,-wi,ldir);
//vec3 Schlick(vec3 F0, float cost){
vec3 F2 = Schlick(vec3(0.3), max(dot(n,-wi),0.));
vec3 specular2 = (D2*G2*F2)/max(4.*max(dot(ldir,n),0.6)*max(dot(-wi,n),0.),0.001);
brdf2 = lerp(brdf2, specular2, probability);
//pdf = lerp(pdf, ggx_pdf(max(dot(reflect(wi,n), d),0.), rough), probability);
brdf2 *= 1.0+(1.0*probability*max(dot(d,n),0.0));
//pdf = max(pdf, 0.0001);




float k2 = 0.;
vec3 lpos = p;
float l2 = 0.;
vec3 c2 = vec3(0.);
float lpdf = 1.0/((2.*3.14159)*(1.0-cos(theta)));

//samplesun(vec3 d, inout uint r, float a){
vec3 possition = p;
	vec3 origin2 = p;

vec3 nnnormal=n; 
vec4[2] voxel2;
int ID2 = 0;
if(i < 6 && i > 0){
//bool hit22 = rayTrace(ldir ,origin2,possition, nnnormal, voxel2, ID2, 128-min(i, 1)*64);

float pdf2 = max(dot(ldir, n),0.)/3.14159;
//pdf2 = lerp(pdf2, );
pdf2 = lerp(pdf2, hmpdf2, probability);
//pdf2 = lerp(pdf2, ggx_pdf(max(dot(reflect(wi,n), ldir),0.), rough), probability);
pdf2 = max(pdf2, 0.000001);
float weigth = powerheuristics(lpdf, pdf2);
float zz = max(dot(vec3(0.,0.,1.), ldir2.xzy),0.);
if(dot(ldir, n) > 0.){

vec3 woo = quickShadow2(VoxelSpaceToSceneSpace(p));
col2 += tt*brdf2*5.*woo*vec3(0.9,0.7,0.5)*max(dot(ldir,n),0.0);

}
}else if(i == 0){

vec3 wo = quickShadow();
float zz = max(dot(vec3(0.,0.,1.), ldir2.xzy),0.);
col += tt*brdf2*wo*5.*vec3(0.9,0.8,0.5)*max(dot(ldir,n),0.0);

}

if(i == 0){
firstcol = col;

}

tt*=brdf*max(dot(d,n),0.)/pdf;
if(i > 3){
float t_max = max(tt.x, max(tt.y, tt.z));
if(t_max < rndf(r)){
break;
}
tt/=t_max;
}



}else{

if(i==0){
//col += tt*skyp3(d, ldir2)*0.7;
distat = 696969.;

vec3 final = skyp3(d.xzy, ldir2.xzy);
float dist = 0.;
vec4 mmm = clouds2(vec3(0.,200.,5500.),d.xzy,ldir2.xzy,dist);
vec3 skys = final;
final = final*mmm.w + mmm.xyz*0.9;
float f = exp(-dist * 0.00048);
final.xyz = final.xyz * f + skys.xyz * (1.0 - f);
float zz = max(dot(vec3(0.,0.,1.), ldir2.xzy),0.);
    final *= max(zz, 0.3)*mix(vec3(1.), vec3(0.3,0.2,0.2), 1.0-zz);
col += tt*final*1.;
firstcol = col;
firstHit = (camera + d*10. );
}else{
	float zz = max(dot(vec3(0.,0.,1.), ldir2.xzy),0.);
col2 += tt*skyp3(d.xzy, ldir2.xzy)*2.0;
}

break;
}

}

//camera = VoxelSpaceToWorldSpace(p);
//vec3 firstHit
/*ec3 cam2 = camera;
vec3 div = (firstHit - camera)/40.;
vec3 accum = vec3(0.);
for(int i = 0; i < 40; i++){
cam2 += div*max(rndf(r),0.6);
accum += quickShadow2(VoxelSpaceToSceneSpace((cam2)));
}

accum /= 30.;
//firstcol += accum*vec3(0.9,0.6,0.2);
distat2 = accum.x;*/
return col2;
}



/***********************************************************************/
	/* Text Rendering */
	const int
		_A    = 0x64bd29, _B    = 0x749d27, _C    = 0xe0842e, _D    = 0x74a527,
		_E    = 0xf09c2f, _F    = 0xf09c21, _G    = 0xe0b526, _H    = 0x94bd29,
		_I    = 0xf2108f, _J    = 0x842526, _K    = 0x9284a9, _L    = 0x10842f,
		_M    = 0x97a529, _N    = 0x95b529, _O    = 0x64a526, _P    = 0x74a4e1,
		_Q    = 0x64acaa, _R    = 0x749ca9, _S    = 0xe09907, _T    = 0xf21084,
		_U    = 0x94a526, _V    = 0x94a544, _W    = 0x94a5e9, _X    = 0x949929,
		_Y    = 0x94b90e, _Z    = 0xf4106f, _0    = 0x65b526, _1    = 0x431084,
		_2    = 0x64904f, _3    = 0x649126, _4    = 0x94bd08, _5    = 0xf09907,
		_6    = 0x609d26, _7    = 0xf41041, _8    = 0x649926, _9    = 0x64b904,
		_APST = 0x631000, _PI   = 0x07a949, _UNDS = 0x00000f, _HYPH = 0x001800,
		_TILD = 0x051400, _PLUS = 0x011c40, _EQUL = 0x0781e0, _SLSH = 0x041041,
		_EXCL = 0x318c03, _QUES = 0x649004, _COMM = 0x000062, _FSTP = 0x000002,
		_QUOT = 0x528000, _BLNK = 0x000000, _COLN = 0x000802, _LPAR = 0x410844,
		_RPAR = 0x221082;

	const ivec2 MAP_SIZE = ivec2(5, 5);

	int GetBit(int bitMap, int index) {
		return (bitMap >> index) & 1;
	}

	float DrawChar(int charBitMap, inout vec2 anchor, vec2 charSize, vec2 uv) {
		uv = (uv - anchor) / charSize;

		anchor.x += charSize.x;

		if (!all(lessThan(abs(uv - vec2(0.5)), vec2(0.5))))
			return 0.0;

		uv *= MAP_SIZE;

		int index = int(uv.x) % MAP_SIZE.x + int(uv.y)*MAP_SIZE.x;

		return GetBit(charBitMap, index);
	}

	const int STRING_LENGTH = 8;
	int[STRING_LENGTH] drawString;

	float DrawString(inout vec2 anchor, vec2 charSize, int stringLength, vec2 uv) {
		uv = (uv - anchor) / charSize;

		anchor.x += charSize.x * stringLength;

		if (!all(lessThan(abs(uv / vec2(stringLength, 1.0) - vec2(0.5)), vec2(0.5))))
			return 0.0;

		int charBitMap = drawString[int(uv.x)];

		uv *= MAP_SIZE;

		int index = int(uv.x) % MAP_SIZE.x + int(uv.y)*MAP_SIZE.x;

		return GetBit(charBitMap, index);
	}

	#define Log10(x) (log2(x) / 3.32192809488)

	float DrawInt(int val, inout vec2 anchor, vec2 charSize, vec2 uv) {
		if (val == 0) return DrawChar(_0, anchor, charSize, uv);

		const int _DIGITS[10] = int[10](_0,_1,_2,_3,_4,_5,_6,_7,_8,_9);

		bool isNegative = val < 0.0;

		if (isNegative) drawString[0] = _HYPH;

		val = abs(val);

		int posPlaces = int(ceil(Log10(abs(val) + 0.001)));
		int strIndex = posPlaces - int(!isNegative);

		while (val > 0) {
			drawString[strIndex--] = _DIGITS[val % 10];
			val /= 10;
		}

		return DrawString(anchor, charSize, posPlaces + int(isNegative), texcoord);
	}

	float DrawFloat(float val, inout vec2 anchor, vec2 charSize, int negPlaces, vec2 uv) {
		int whole = int(val);
		int part  = int(fract(abs(val)) * pow(10, negPlaces));

		int posPlaces = max(int(ceil(Log10(abs(val)))), 1);

		anchor.x -= charSize.x * (posPlaces + int(val < 0) + 0.25);
		float ret = 0.0;
		ret += DrawInt(whole, anchor, charSize, uv);
		ret += DrawChar(_FSTP, anchor, charSize, texcoord);
		anchor.x -= charSize.x * 0.3;
		ret += DrawInt(part, anchor, charSize, uv);

		return ret;
	}

	void DrawDebugText() {
		vec2 charSize = vec2(0.018) * viewResolution.yy / viewResolution;
		vec2 texPos = vec2(charSize.x * 2.75, 1.0 - charSize.y * 1.2);

		if ( texcoord.x > charSize.x * 18.0
		||   texcoord.y < 1 - charSize.y * 4.0)
		{ return; }

		vec3 color = vec3(0.0);

        texPos.y -= charSize.y * 2.0 - 0.0045;

		drawString = int[STRING_LENGTH](_F,_R,_M,_T,_I,_M,_E,_COLN);
		color += DrawString(texPos, charSize, 8, texcoord);
		texPos.x += charSize.x * 3.0;
		color += DrawFloat(frameTime * 1000, texPos, charSize, 5, texcoord);
		//outColor = color;
	}    

    #define raytrace

	void main() {
        #ifdef raytrace
uint r = uint(uint(texcoord.x*1000.) * uint(1973) 
    + uint(texcoord.y * 1000.) * uint(9277) 
    + uint(frameCounter) * uint(26699)) | uint(1);

		vec4[2] voxel; vec3 hitPos = cameraPosition;vec3 hitNormal;
		vec4 p22   = vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);
		vec3 dir = (gbufferProjectionInverse * p22).xyz / (gbufferProjectionInverse * p22).w;
		     dir = normalize(mat3(gbufferModelViewInverse) * dir);


		//bool hit = rayTrace(dir,hitPos, hitNormal, voxel);

		//float Depth = texture2D(depthtex0, texcoord).w;

	//vec3 sunPos = mat3(gbufferModelViewInverse) * normalize(sunPosition);
	//vec3 coll = getMaterial(hitPos, hitNormal, voxel).albedo;
   
		//vec3 renderpixel(vec3 p, vec3 d, inout uint r, float check, vec3 noblur){
float distat = 0.0;
vec3 prevp = vec3(0.);
vec3 prevcol = vec3(0.);
float spec = 0.;
vec3 firstnorm = vec3(0.);
vec3 prevcol2 = vec3(0.);
float distat2 = 0.;
		vec3 colorr = renderpixel(SceneSpaceToVoxelSpace(gbufferModelViewInverse[3].xyz), dir, r, distat, prevp, prevcol, spec, firstnorm, prevcol2, distat2);
colorr = clamp(colorr, 0.0, 1.0);

		if(distat != 696969.0){

			//colorr *=1.5;
		//outColor = colorr;
		}else{
		colorr = pow(colorr, vec3(1.0));
       colorr = ACESFilm(colorr);
        colorr = pow(colorr, vec3(1./2.2));
		//outColor = colorr;
		//outColor3 = vec4(colorr,distat);
		}
				//outColor = colorr;

		//outColor3 = vec4(colorr,distat);

		gl_FragData[0] = vec4(normalize(firstnorm), distat);
		gl_FragData[1] = vec4((colorr),distat);
		gl_FragData[2] = vec4(texture2D(colortex2, texcoord));
		gl_FragData[3] = vec4(texture2D(colortex3, texcoord));

		gl_FragData[5] = vec4((prevcol), (distat != 696969.0)?spec:20.5);
		gl_FragData[6] = vec4((prevcol2), distat);


        #endif
        //DrawDebugText();
        //outColor = vec3(getTileSize(voxel) > 16 ? 1 : 0);
	}

#endif
