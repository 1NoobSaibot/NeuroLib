#ifndef RELU_H
#define RELU_H

extern "C" __declspec(dllexport) void ReLU(const float* d_input, float* d_output, int size);
extern "C" __declspec(dllexport) void ReLU_Derivative(
	const float* d_lossVector,
	const float* d_zVector,
	float* d_output,
	int size
);

#endif
