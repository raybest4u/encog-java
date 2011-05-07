__kernel void calculateLayer(
	__global const float *weights,
	__global const int *param,
	__global float *layerOutput)
{
	int gid = get_global_id(0);
	
	const int startIndex = param[0];
	const int outputIndex = param[1]; 
	const int inputIndex = param[2];
	const int inputSize = param[3];

	int index = startIndex;
	const int limitY = inputIndex + inputSize;

	int x = outputIndex + gid;

	float sum = 0;
	for (int y = inputIndex; y < limitY; y++) {
		sum += weights[index++] * layerOutput[y];
	}
	layerOutput[x] = sum;
}