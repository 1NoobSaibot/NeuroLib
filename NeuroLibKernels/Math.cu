#include "Math.cuh"
#include <device_launch_parameters.h>

const int THREADS_PER_BLOCK = 256;

__global__ void vector_sub_f_kernel(const float* d_v1, const float* d_v2, float* d_output, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		d_output[idx] = d_v1[idx] - d_v2[idx];
	}
}

extern "C" void VectorSubF(const float* d_v1, const float* d_v2, float* d_output, int size) {
	int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	vector_sub_f_kernel <<<blocksPerGrid, THREADS_PER_BLOCK >>> (d_v1, d_v2, d_output, size);
}