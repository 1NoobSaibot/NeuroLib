using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuroLib.Libs
{
	internal static class CudaLib
	{
		private const string CUDA_DLL = "cudart64_12.dll";


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void TransferFromHostToDevice(float[] hostArray, IntPtr deviceArray)
		{
			cudaMemcpy(
				deviceArray,
				hostArray,
				(uint)hostArray.Length * sizeof(float),
				cudaMemcpyKind.HostToDevice
			);
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void TransferFromDeviceToHost(IntPtr deviceArray, float[] hostArray)
		{
			cudaMemcpy(
				hostArray,
				deviceArray,
				(uint)hostArray.Length * sizeof(float),
				cudaMemcpyKind.DeviceToHost
			);
		}


		[DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
		public static extern int cudaMalloc(out IntPtr devPtr, uint size);

		[DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
		public static extern int cudaFree(IntPtr devPtr);

		[DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
		public static extern int cudaMemcpy(IntPtr dst, IntPtr src, uint count, cudaMemcpyKind kind);

		[DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
		public static extern int cudaMemcpy(IntPtr dst, float[] src, uint count, cudaMemcpyKind kind);

		[DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
		public static extern int cudaMemcpy(float[] dst, IntPtr src, uint count, cudaMemcpyKind kind);
	}


	public enum cudaMemcpyKind
	{
		HostToDevice = 1,
		DeviceToHost = 2,
		DeviceToDevice = 3,
		HostToHost = 4
	}
}
