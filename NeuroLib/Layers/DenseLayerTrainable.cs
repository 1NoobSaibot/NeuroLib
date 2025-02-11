using NeuroLib.Libs;
using NeuroLib.Math;

namespace NeuroLib.Layers
{
	public class DenseLayerTrainable : DenseLayer
	{
		private readonly CudaDeviceMatrix _sumZ;
		public override CudaDeviceMatrix Sum_Z => _sumZ;
		public readonly CudaDeviceMatrix OutputDeltas;
		public readonly CudaDeviceMatrix InputDeltas;
		private readonly ILayer _inputLayer;


		public DenseLayerTrainable(ILayer input, int outputVectorLength)
			: base(input, outputVectorLength)
		{
			_sumZ = CudaDeviceMatrix.CreateWithShapeOf(Outputs);
			OutputDeltas = CudaDeviceMatrix.CreateWithShapeOf(Outputs);

			if (input is InputLayer)
			{
				InputDeltas = CudaDeviceMatrix.CreateWithShapeOf(Inputs);
			}
			else if (input is DenseLayerTrainable dst_InputLayer)
			{
				InputDeltas = dst_InputLayer.OutputDeltas;
			}
			else
			{
				throw new ArgumentException("Input layer must be either InputLayer or DenseLayerTrainable");
			}

			_inputLayer = input;
		}


		public void StartBackward(CudaDeviceMatrix batchOfExpectedOutputs)
		{
			NeuroLibKernels.VectorSubF(batchOfExpectedOutputs, Outputs, OutputDeltas);
			Backward();
		}


		protected void Backward()
		{
			Activation.ApplyDerivative(OutputDeltas, Sum_Z, OutputDeltas);
			NeuroLibKernels.DenseLayer_Backward(
				batch_of_input_vectors: Inputs.DeviceMemoryPointer,
				weights_and_biases: WeightsAndBiases.DeviceMemoryPointer,
				batch_of_output_delta_vectors: OutputDeltas.DeviceMemoryPointer,
				batch_of_input_delta_vectors: InputDeltas.DeviceMemoryPointer,
				learning_rate: 0.05f,
				input_length: Inputs.Columns,
				output_length: Outputs.Columns,
				batch_size: Inputs.Rows
			);

			if (_inputLayer is DenseLayerTrainable dst_InputLayer)
			{
				dst_InputLayer.Backward();
			}
		}


		public void RandomiseWeights(Random rnd)
		{
			float[] weights = new float[WeightsAndBiases.ElementCount];
			for (int i = 0; i < weights.Length; i++)
			{
				weights[i] = rnd.NextSingle();
			}
			WeightsAndBiases.SetValues(weights);
		}
	}
}
