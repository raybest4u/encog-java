__kernel void calculateLayer(
	__global const float *weights,
	__global const int *param,
	__global const float *layerOutput,
	__global float *output)
{
	int gid = get_global_id(0);
	
	const int startIndex = param[0];
	const int outputIndex = param[1]; 
    const int outputSize = param[2];
	const int inputIndex = param[3];
	const int inputSize = param[4];
	
	int index = startIndex+(inputSize*gid);
	const int limitY = inputIndex + inputSize;

	float sum = 0;
	for (int y = inputIndex; y < limitY; y++) {
		sum += weights[index++] * layerOutput[y];
	}
	output[gid] = sum;
}