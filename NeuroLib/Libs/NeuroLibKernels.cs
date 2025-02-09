using NeuroLib.Math;
using System.Runtime.InteropServices;

namespace NeuroLib.Libs
{
	internal static class NeuroLibKernels
	{
		private const string DLL = "NeuroLibKernels.dll";


		[DllImport(DLL, EntryPoint = "DenseLayer_ForwardSum")]
		public static extern void DenseLayer_ForwardSum(
			IntPtr input,
			IntPtr weights,
			IntPtr output,
			int inputSize,
			int outputSize,
			int batchSize
		);


		public static void DenseLayer_ForwardSum(
			CudaDeviceMatrix input,
			CudaDeviceMatrix weights,
			CudaDeviceMatrix output
		)
		{
			int inputSize = input.Columns;
			int batchSize = input.Rows;

			if (weights.Rows != (inputSize + 1))
			{
				throw new ArgumentException("Weights matrix has invalid rows count");
			}
			int outputSize = weights.Columns;

			if (output.Rows != batchSize)
			{
				throw new ArgumentException("Output matrix has invalid rows count");
			}
			if (output.Columns != outputSize)
			{
				throw new ArgumentException("Output matrix has invalid columns count");
			}

			DenseLayer_ForwardSum(
				input.DeviceMemoryPointer,
				weights.DeviceMemoryPointer,
				output.DeviceMemoryPointer,
				inputSize,
				outputSize,
				batchSize
			);
		}
	}
}
