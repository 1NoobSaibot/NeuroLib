#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void dl_forward_sum(
	const float* __restrict__ input_dptr,
	const float* __restrict__ weights_dptr,
	float* __restrict__ output_dptr,
	int input_length,
	int output_length,
	int batch_size
)
{
	int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if (batch_idx >= batch_size)
	{
		return;
	}

	int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (output_idx >= output_length)
	{
		return;
	}

	float sum = weights_dptr[output_length * input_length +  output_idx];
	for (int input_idx = 0; input_idx < input_length; input_idx++)
	{
		float input_value = input_dptr[input_length * batch_idx + input_idx];
		float weight_Value = weights_dptr[output_length * input_idx + output_idx];
		sum += input_value * weight_Value;
	}

	output_dptr[output_length * batch_idx + output_idx] = sum;
}


extern "C" __declspec(dllexport) void DenseLayer_ForwardSum(
	const float* input_dptr,
	const float* weights_dptr,
	float* output_dptr,
	int input_length,
	int output_length,
	int batch_size
)
{
	dim3 blockDim(16, 16);
	dim3 gridDim(
		(output_length + blockDim.x - 1) / blockDim.x,
		(batch_size + blockDim.y - 1) / blockDim.y
	);

	dl_forward_sum <<<gridDim, blockDim>>> (
		input_dptr,
		weights_dptr,
		output_dptr,
		input_length,
		output_length,
		batch_size
	);
}
