__kernel void simpleKernel(
	__global const float *a,
	__global const float *b,
	__global float *c)
{
	int gid = get_global_id(0);
	c[gid] = gid;
	//c[gid] = a[gid] * b[gid];
}