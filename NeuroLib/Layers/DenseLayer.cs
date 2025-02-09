using NeuroLib.Libs;
using NeuroLib.Math;

namespace NeuroLib.Layers
{
	public class DenseLayer
	{
		public readonly CudaDeviceMatrix Inputs;
		public readonly CudaDeviceMatrix WeightsAndBiases;
		public readonly CudaDeviceMatrix Outputs;


		public DenseLayer(CudaDeviceMatrix input, int outputVectorLength)
		{
			if (outputVectorLength < 1)
			{
				throw new ArgumentException("Output length must be greater than 0");
			}

			Inputs = input;
			int inputVectorLength = input.Columns;
			int batchSize = input.Rows;

			WeightsAndBiases = new CudaDeviceMatrix(rows: inputVectorLength + 1, columns: outputVectorLength);
			Outputs = new CudaDeviceMatrix(rows: batchSize, columns: outputVectorLength);
		}


		public void Forward()
		{
			NeuroLibKernels.DenseLayer_ForwardSum(Inputs, WeightsAndBiases, Outputs);
		}
	}
}
