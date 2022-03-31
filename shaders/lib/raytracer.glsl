// Primitives
//Modified code from Lith https://cdn.discordapp.com/attachments/276979724922781697/862415868934619206/Voxelization_Geometry.zip

float maxof(vec3 x) { return max(x.x, max(x.y, x.z)); }
float minof(vec3 x) { return min(x.x, min(x.y, x.z)); }

bool IntersectAABB(vec3 pos, vec3 dir, vec3 minBounds, vec3 maxBounds) {
	vec3 minBoundsDist = ((minBounds - pos) / dir);
	vec3 maxBoundsDist = ((maxBounds - pos) / dir);
	
	vec3 minDists = min(minBoundsDist, maxBoundsDist);
	vec3 maxDists = max(minBoundsDist, maxBoundsDist);
	
	ivec3 a = floatBitsToInt(minDists.xxy - maxDists.yzx);
	ivec3 b = floatBitsToInt(minDists.yzz - maxDists.zxy);
	a = a & b;
	return (a.x & a.y & a.z) < 0;
}


bool IntersectBlock2(vec3 pos, vec3 origin, vec3 direction, vec3 minBounds, vec3 maxBounds ,out vec3 hitPos, inout vec3 hitNormal) {
	vec3 minBoundsDist = ((minBounds - pos) / direction);
	vec3 maxBoundsDist = ((maxBounds - pos) / direction);

	vec3 positiveDir = step(0.0, direction);
	vec3 dists       = mix(maxBoundsDist, minBoundsDist, positiveDir);

	float dist;
	if (dists.x > dists.y) {
		if (dists.x > dists.z) {
			dist = dists.x;
			hitNormal = vec3(-positiveDir.x * 2.0 + 1.0, 0.0, 0.0);
		} else {
			dist = dists.z;
			hitNormal = vec3(0.0, 0.0, -positiveDir.z * 2.0 + 1.0);
		}
	} else if (dists.y > dists.z) {
		dist = dists.y;
		hitNormal = vec3(0.0, -positiveDir.y * 2.0 + 1.0, 0.0);
	} else {
		dist = dists.z;
		hitNormal = vec3(0.0, 0.0, -positiveDir.z * 2.0 + 1.0);
	}

	hitPos = dist * direction + origin;

	return dist > 0.0;
}


bool IntersectBlock(vec3 origin, ivec3 index, vec3 direction, out vec3 hitPos, inout vec3 hitNormal) {
	vec3 minBoundsDist = (      index - origin) / direction;
	vec3 maxBoundsDist = (1.0 + index - origin) / direction;

	vec3 positiveDir = step(0.0, direction);
	vec3 dists       = mix(maxBoundsDist, minBoundsDist, positiveDir);

	float dist;
	if (dists.x > dists.y) {
		if (dists.x > dists.z) {
			dist = dists.x;
			hitNormal = vec3(-positiveDir.x * 2.0 + 1.0, 0.0, 0.0);
		} else {
			dist = dists.z;
			hitNormal = vec3(0.0, 0.0, -positiveDir.z * 2.0 + 1.0);
		}
	} else if (dists.y > dists.z) {
		dist = dists.y;
		hitNormal = vec3(0.0, -positiveDir.y * 2.0 + 1.0, 0.0);
	} else {
		dist = dists.z;
		hitNormal = vec3(0.0, 0.0, -positiveDir.z * 2.0 + 1.0);
	}

	hitPos = dist * direction + origin;

	return dist > 0.0;
}

bool IntersectVoxel(vec3 origin, ivec3 index, vec3 direction, int id, out vec3 hitPos, inout vec3 hitNormal) {
	// default value
	hitPos = origin;

	 
	return IntersectBlock(origin, index, direction, hitPos, hitNormal);
}

 