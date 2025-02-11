using NeuroLib.Libs;
using NeuroLib.Math;

namespace NeuroLib.Activations
{
	public class NoActivation : IActivationFunction
	{
		public static IActivationFunction Instance { get; } = new NoActivation();


		public void Apply(CudaDeviceMatrix input, CudaDeviceMatrix output)
		{
			if (object.ReferenceEquals(input, output))
			{
				// No need to do anything, it's the same matrix
				return;
			}

			// But if not the same, just transfer the data
			CudaLib.cudaMemcpy(
				output.DeviceMemoryPointer,
				input.DeviceMemoryPointer,
				(uint)(input.ElementCount * sizeof(float)),
				cudaMemcpyKind.DeviceToDevice
			);
		}

		public void ApplyDerivative(CudaDeviceMatrix loss, CudaDeviceMatrix z, CudaDeviceMatrix result)
		{
			if (object.ReferenceEquals(loss, result))
			{
				return;
			}

			CudaLib.cudaMemcpy(
				dst: result.DeviceMemoryPointer,
				src: loss.DeviceMemoryPointer,
				count: (uint)(loss.ElementCount * sizeof(float)),
				kind: cudaMemcpyKind.DeviceToDevice
			);
		}

		private NoActivation()
		{ }
	}
}
