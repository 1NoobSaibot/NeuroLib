#include "ReLU.cuh"
#include <device_launch_parameters.h>


__global__ void ReLU_Kernel(const float* input, float* output, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		output[idx] = input[idx] < 0 ? 0 : input[idx];
	}
}


__global__ void ReLU_Derivative_Kernel(
	const float* d_lossVector,
	const float* d_zVector,
	float* d_output,
	int size
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		d_output[idx] = d_lossVector[idx] * static_cast<float>(d_zVector[idx] >= 0);
	}
}


extern "C" void ReLU(const float* d_input, float* d_output, int size) {
	const int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	ReLU_Kernel <<<blocksPerGrid, threadsPerBlock >>> (d_input, d_output, size);
}


extern "C" void ReLU_Derivative(
	const float* d_lossVector,
	const float* d_zVector,
	float* d_output,
	int size
) {
	const int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	ReLU_Derivative_Kernel <<<blocksPerGrid, threadsPerBlock>>> (d_lossVector, d_zVector, d_output, size);
}
