using NeuroLib.Activations;
using NeuroLib.Libs;
using NeuroLib.Math;

namespace NeuroLib.Layers
{
	public class DenseLayer : ILayer
	{
		private ILayer? NextLayer;
		public readonly CudaDeviceMatrix Inputs;
		public readonly CudaDeviceMatrix WeightsAndBiases;
		public CudaDeviceMatrix Outputs { get; }
		public readonly IActivationFunction Activation = new ReLU();
		public virtual CudaDeviceMatrix Sum_Z => Outputs;


		public DenseLayer(ILayer inputLayer, int outputVectorLength)
		{
			if (outputVectorLength < 1)
			{
				throw new ArgumentException("Output length must be greater than 0");
			}

			Inputs = inputLayer.Outputs;
			inputLayer.ApplyNext(this);
			int inputVectorLength = Inputs.Columns;
			int batchSize = Inputs.Rows;

			WeightsAndBiases = new CudaDeviceMatrix(rows: inputVectorLength + 1, columns: outputVectorLength);
			Outputs = new CudaDeviceMatrix(rows: batchSize, columns: outputVectorLength);
		}


		public void Forward()
		{
			NeuroLibKernels.DenseLayer_ForwardSum(Inputs, WeightsAndBiases, Sum_Z);
			Activation.Apply(Sum_Z, Outputs);
			NextLayer?.Forward();
		}


		public void ApplyNext(ILayer nextLayer)
		{
			if (NextLayer is not null)
			{
				throw new ArgumentException("A next layer is already aplied");
			}
			NextLayer = nextLayer;
		}
	}
}
