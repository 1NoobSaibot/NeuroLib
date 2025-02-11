using NeuroLib.Math;

namespace NeuroLib.Activations
{
	public interface IActivationFunction
	{
		void Apply(CudaDeviceMatrix input, CudaDeviceMatrix output);
		void ApplyDerivative(CudaDeviceMatrix loss, CudaDeviceMatrix z, CudaDeviceMatrix result);
	}
}
