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

		[DllImport(DLL, EntryPoint = "DenseLayer_Backward")]
		public static extern void DenseLayer_Backward(
			IntPtr batch_of_input_vectors,
			IntPtr weights_and_biases,
			IntPtr batch_of_output_delta_vectors,
			IntPtr batch_of_input_delta_vectors,
			float learning_rate,
			int input_length,
			int output_length,
			int batch_size
		);


		[DllImport(DLL, EntryPoint = "ReLU")]
		public static extern void ReLU(IntPtr devInput, IntPtr devOutput, int size);


		/// <summary>
		/// output = Loss * ReLu_Derivave(Z)
		/// </summary>
		[DllImport(DLL, EntryPoint = "ReLU_Derivative")]
		public static extern void ReLU_Derivative(IntPtr devLoss, IntPtr devZ, IntPtr devOutput, int size);


		/// <summary>
		/// C = A - B; All vectors must be the same size
		/// </summary>
		/// <param name="size"></param>
		[DllImport(DLL, EntryPoint = "VectorSubF")]
		private static extern void _vectorSubF(IntPtr devA, IntPtr devB, IntPtr devC, int size);


		public static void VectorSubF(CudaDeviceMatrix a, CudaDeviceMatrix minusB, CudaDeviceMatrix result)
		{
			if (a.ElementCount != minusB.ElementCount || a.ElementCount != result.ElementCount)
			{
				throw new ArgumentException("Input vectors must have the same size");
			}
			_vectorSubF(
				a.DeviceMemoryPointer,
				minusB.DeviceMemoryPointer,
				result.DeviceMemoryPointer,
				a.ElementCount
			);
		}


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
