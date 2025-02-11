using NeuroLib.Libs;
using System.Runtime.CompilerServices;

namespace NeuroLib.Math
{
	public enum MatrixLayout
	{
		RowMajor,
		ColumnMajor
	}


	public class CudaDeviceMatrix
	{
		public IntPtr DeviceMemoryPointer { get; }
		public int Rows { get; }
		public int Columns { get; }
		public const MatrixLayout DEFAULT_CUDA_MATRIX_LAYOUT = MatrixLayout.ColumnMajor;
		public MatrixLayout Layout => DEFAULT_CUDA_MATRIX_LAYOUT;

		public int ElementCount => Rows * Columns;
		public int SizeInBytes => ElementCount * sizeof(float);


		public CudaDeviceMatrix(
			int rows,
			int columns
		)
		{
			if (rows <= 0 || columns <= 0)
			{
				throw new ArgumentException("Matrix dimensions must be positive.");
			}

			Rows = rows;
			Columns = columns;

			IntPtr ptr = IntPtr.Zero;
			var result = CudaLib.cudaMalloc(out ptr, (uint)SizeInBytes);
			if (result != 0)
			{
				throw new Exception("CUDA malloc failed: code=" + result);
			}
			DeviceMemoryPointer = ptr;
		}


		~CudaDeviceMatrix()
		{
			CudaLib.cudaFree(DeviceMemoryPointer);
		}


		public void SetValues(float[] hostData)
		{
			if (hostData.Length != ElementCount)
			{
				throw new ArgumentException("Input array size does not match matrix size.");
			}

			CudaLib.TransferFromHostToDevice(hostData, DeviceMemoryPointer);
		}


		public void GetValues(float[] hostData)
		{
			if (hostData.Length != ElementCount)
				throw new ArgumentException("Output array size does not match matrix size.");

			CudaLib.TransferFromDeviceToHost(DeviceMemoryPointer, hostData);
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static CudaDeviceMatrix CreateWithShapeOf(CudaDeviceMatrix proto)
		{
			return new CudaDeviceMatrix(rows: proto.Rows, columns: proto.Columns);
		}
	}
}
