#include "ReLU.cuh"
#include "Math.cuh"
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


__global__ void dl_backward(
	const float* batch_of_input_vectors,
	float* weights_and_biases,
	const float* batch_of_output_delta_vectors,
	float* batch_of_input_delta_vectors,
	const float learning_rate,
	const int input_length,
	const int output_length,
	const int batch_size
) {
	int inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int outputIdx = blockIdx.y * blockDim.y + threadIdx.y;

	if (inputIdx > input_length || outputIdx >= output_length)
	{
		return;
	}

	int weightIdx = output_length * inputIdx + outputIdx;
	float oldWeightValue = weights_and_biases[weightIdx];
	float weightDelta = 0;
	for (int batchIdx = 0; batchIdx < batch_size; batchIdx++)
	{
		float outputDeltaValue = batch_of_output_delta_vectors[output_length * batchIdx + outputIdx];
		int inputBatchIdx = input_length * batchIdx + inputIdx;
		float inputValue = 1;

		if (inputIdx < input_length)
		{
			// We are processing not a bias
			inputValue = batch_of_input_vectors[inputBatchIdx];
			batch_of_input_delta_vectors[inputBatchIdx] = oldWeightValue * outputDeltaValue;
		}
		
		weightDelta += learning_rate * inputValue * outputDeltaValue;
	}
	weights_and_biases[weightIdx] = oldWeightValue + weightDelta;
}


extern "C" __declspec(dllexport) void DenseLayer_Backward(
	const float* batch_of_input_vectors,
	float* weights_and_biases,
	const float* batch_of_output_delta_vectors,
	float* batch_of_input_delta_vectors,
	float learning_rate,
	int input_length,
	int output_length,
	int batch_size
) {
	dim3 blockDim(16, 16);
	dim3 gridDim(
		((input_length + 1) + blockDim.x - 1) / blockDim.x,
		(output_length + blockDim.y - 1) / blockDim.y
	);

	dl_backward <<<gridDim, blockDim >>> (
		batch_of_input_vectors,
		weights_and_biases,
		batch_of_output_delta_vectors,
		batch_of_input_delta_vectors,
		learning_rate,
		input_length,
		output_length,
		batch_size
	);
}
