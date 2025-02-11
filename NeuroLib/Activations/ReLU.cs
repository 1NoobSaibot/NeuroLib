using NeuroLib.Libs;
using NeuroLib.Math;

namespace NeuroLib.Activations
{
	public class ReLU : IActivationFunction
	{
		public void Apply(CudaDeviceMatrix input, CudaDeviceMatrix output)
		{
			NeuroLibKernels.ReLU(
				input.DeviceMemoryPointer,
				output.DeviceMemoryPointer,
				input.ElementCount
			);
		}


		public void ApplyDerivative(CudaDeviceMatrix loss, CudaDeviceMatrix z, CudaDeviceMatrix result)
		{
			NeuroLibKernels.ReLU_Derivative(
				devLoss: loss.DeviceMemoryPointer,
				devZ: z.DeviceMemoryPointer,
				devOutput: result.DeviceMemoryPointer,
				size: loss.ElementCount
			);
		}
	}
}
