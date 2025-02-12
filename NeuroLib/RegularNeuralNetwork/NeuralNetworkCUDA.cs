using NeuroLib.Layers;
using NeuroLib.Math;


namespace NeuroLib
{
	public class NeuralNetworkCUDA
	{
		private readonly ILayer[] _layers;
		public int AmountOfLayers => _layers.Length;
		private readonly InputLayer _inputLayer;
		private readonly DenseLayerTrainable _outputLayer;
		public readonly int BatchSize;


		public NeuralNetworkCUDA(int batchSize, int[] amountsOfNeurons)
		{
			if (batchSize < 1)
			{
				throw new ArgumentException($"must be a positive number", nameof(batchSize));
			}
			if (amountsOfNeurons.Length < 2)
			{
				throw new ArgumentException("Neural network cannot have less than two layers of neurons");
			}

			BatchSize = batchSize;
			_layers = new ILayer[amountsOfNeurons.Length];
			_inputLayer = new InputLayer(amountsOfNeurons[0], batchSize);
			_layers[0] = _inputLayer;
			for (int i = 1; i < _layers.Length; i++)
			{
				_layers[i] = new DenseLayerTrainable(_layers[i - 1], amountsOfNeurons[i]);
			}

			_outputLayer = (DenseLayerTrainable)_layers[^1];
		}


		public void Forward(float[] input, float[] output)
		{
			if (input.Length != _inputLayer.Inputs.ElementCount)
			{
				throw new ArgumentException("Length must be equal to product of inputLength * batchSize", nameof(input));
			}
			if (output.Length != _outputLayer.Outputs.ElementCount)
			{
				throw new ArgumentException("Length must be equal to product of outputLength * batchSize", nameof(output));
			}

			_inputLayer.Inputs.SetValues(input);
			_inputLayer.Forward();
			_outputLayer.Outputs.GetValues(output);
		}


		public void Randomize(Random rnd)
		{
			for (int i = 1; i < _layers.Length; i++)
			{
				DenseLayerTrainable layer = (DenseLayerTrainable)_layers[i];
				layer.RandomiseWeights(rnd);
			}
		}


		public void Fit(Random rnd, float[][] inputDataSet, float[][] trueOutValues, int epochs)
		{
			if (inputDataSet.Length != trueOutValues.Length)
			{
				throw new ArgumentException("Lengthes of data sets must be equal to each other");
			}
			if (inputDataSet.Length < BatchSize)
			{
				throw new ArgumentException("input data sets");
			}
			int inputLength = _inputLayer.Inputs.Columns;
			int outputLength = _outputLayer.Outputs.Columns;
			foreach (var inputVector in inputDataSet)
			{
				if (inputVector.Length != inputLength)
				{
					throw new ArgumentException($"One or more items has invalid length", nameof(inputDataSet));
				}
			}
			foreach (var outputVector in trueOutValues)
			{
				if (outputVector.Length != outputLength)
				{
					throw new ArgumentException($"One or more items has invalid length", nameof(trueOutValues));
				}
			}

			int[] dataSetOrder = CreateDataSetOrderArray(inputDataSet.Length);
			float[] inputBatchBuffer = new float[_inputLayer.Inputs.ElementCount];
			float[] outputBatchBuffer = new float[_outputLayer.Outputs.ElementCount];

			CudaDeviceMatrix trueOut = CudaDeviceMatrix.CreateWithShapeOf(_outputLayer.Outputs);
			for (int epoch = 0; epoch < epochs; epoch++)
			{
				rnd.Shuffle(dataSetOrder);
				for (int batchIndex = 0; batchIndex < inputDataSet.Length; batchIndex += BatchSize)
				{
					for (int itemIndex = 0; itemIndex < BatchSize; itemIndex++)
					{
						int index = dataSetOrder[batchIndex + itemIndex];
						float[] inputVector = inputDataSet[index];
						float[] trueOutputVector = trueOutValues[index];

						Array.Copy(inputVector, 0, inputBatchBuffer, itemIndex * inputLength, inputVector.Length);
						Array.Copy(trueOutputVector, 0, outputBatchBuffer, itemIndex * outputLength, trueOutputVector.Length);
					}

					_inputLayer.Inputs.SetValues(inputBatchBuffer);
					_inputLayer.Forward();
					trueOut.SetValues(outputBatchBuffer);
					_outputLayer.StartBackward(trueOut);
				}
			}


			static int[] CreateDataSetOrderArray(int length)
			{
				int[] order = new int[length];
				int index = 0;
				while (index < length)
				{
					order[index] = index;
					++index;
				}
				return order;
			}
		}


		public int GetNumberOfUnitsForLayer(int i)
		{
			ILayer layer = _layers[i];
			if (layer is InputLayer il)
			{
				return il.Inputs.Columns;
			}
			if (layer is DenseLayer dlt)
			{
				return dlt.Outputs.Columns;
			}
			throw new Exception();
		}


		public float[] GetWeightsOfLayer(int i)
		{
			ILayer layer = _layers[i];
			if (layer is DenseLayer dl)
			{
				return dl.WeightsAndBiases.GetValues();
			}
			throw new Exception();
		}


		public void SetWeightsOfLayer(int i, float[] weights)
		{
			ILayer layer = _layers[i];
			if (layer is DenseLayer dl)
			{
				dl.WeightsAndBiases.SetValues(weights);
			}
			else
			{
				throw new Exception();
			}
		}
	}
}
