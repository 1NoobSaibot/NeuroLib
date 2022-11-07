using NeuroLib.Helpers;
using System;
using System.Diagnostics;


namespace NeuroLib
{
	public class NeuralNetwork
	{
		private float[][,] _weights;
		private float[][] _layerOutputs;
		private float[][] _biases;


		public int AmountOfLayers => _layerOutputs.Length;

		
		public NeuralNetwork(params int[] amountsOfNeurons)
		{
			Debug.Assert(amountsOfNeurons.Length >= 2);

			_CreateLayerVectors(amountsOfNeurons);
			_CreateWeightsMatrixes(amountsOfNeurons);
			_CreateBiasVectors(amountsOfNeurons);
		}


		public void SetInput(int index, float value)
		{
			_layerOutputs[0][index] = value;
		}


		public void SetInputs(float[] values)
		{
			int inputLength = _layerOutputs[0].Length;
			for (int i = 0; i < inputLength; i++)
			{
				_layerOutputs[0][i] = values[i];
			}
		}


		public void Tick()
		{
			for (int outLayerIndex = 1; outLayerIndex < _layerOutputs.Length; outLayerIndex++)
			{
				float[] outLayer = _layerOutputs[outLayerIndex];
				float[] inLayer = _layerOutputs[outLayerIndex - 1];
				float[] biasVector = _biases[outLayerIndex - 1];
				float[,] weightMatrix = _weights[outLayerIndex - 1];

				for (int outNeuronIndex = 0; outNeuronIndex < outLayer.Length; outNeuronIndex++)
				{
					outLayer[outNeuronIndex] = biasVector[outNeuronIndex];

					for (int inNeuronIndex = 0; inNeuronIndex < inLayer.Length; inNeuronIndex++)
					{
						outLayer[outNeuronIndex] += inLayer[inNeuronIndex] * weightMatrix[inNeuronIndex, outNeuronIndex];
					}

					outLayer[outNeuronIndex] = _Sigmoid(outLayer[outNeuronIndex]);
				}
			}


			float _Sigmoid(float input)
			{
				return 1 / (1 + (float)Math.Exp(-4 * input));
			}
		}


		public float GetOutput(int outIndex)
		{
			int outLayerIndex = _layerOutputs.Length - 1;
			return _layerOutputs[outLayerIndex][outIndex];
		}


		public float[,] GetWeightMatrix(int outLayerIndex)
		{
			return _weights[outLayerIndex - 1];
		}


		public float[] GetBiasVector(int layerIndex)
		{
			return _biases[layerIndex - 1];
		}


		public NeuralNetwork Clone()
		{
			int[] layers = GetConstructorParams();
			NeuralNetwork copy = new NeuralNetwork(layers);

			for (int i = 1; i < layers.Length; i++)
			{
				float[] dstBiasVector = copy.GetBiasVector(i);
				float[] srcBiasVector = this.GetBiasVector(i);
				float[,] dstWeightMatrix = copy.GetWeightMatrix(i);
				float[,] srcWeightMatrix = this.GetWeightMatrix(i);

				for (int j = 0; j < dstBiasVector.Length; j++)
				{
					dstBiasVector[j] = srcBiasVector[j];

					for (int inNeuron = 0; inNeuron < dstWeightMatrix.GetLength(0); inNeuron++)
					{
						dstWeightMatrix[inNeuron, j] = srcWeightMatrix[inNeuron, j];
					}
				}
			}

			return copy;
		}


		public int[] GetConstructorParams()
		{
			int[] layers = new int[_layerOutputs.Length];
			for (int i = 0; i < layers.Length; i++)
			{
				layers[i] = _layerOutputs[i].Length;
			}
			return layers;
		}


		public void RemoveNeuron(int layer, int neuron)
		{
			if (layer == 0)
			{
				RemoveInputNeuron(neuron);
				return;
			}
			else if (layer == _layerOutputs.Length - 1)
			{
				RemoveOutputNeuron(neuron);
				return;
			}

			_RemoveBiasForNeuron(layer, neuron);
			_layerOutputs[layer] = _layerOutputs[layer].RemoveElementAtIndex(neuron);
			_RemoveInWeights(layer, neuron);
			_RemoveOutWeights(layer, neuron);
		}


		public void RemoveInputNeuron(int neuron)
		{
			_layerOutputs[0] = _layerOutputs[0].RemoveElementAtIndex(neuron);
			_RemoveOutWeights(0, neuron);
		}


		public void RemoveOutputNeuron(int neuron)
		{
			int outLayer = _layerOutputs.Length - 1;
			_layerOutputs[outLayer] = _layerOutputs[outLayer].RemoveElementAtIndex(neuron);
			_RemoveBiasForNeuron(outLayer, neuron);
			_RemoveInWeights(outLayer, neuron);
		}


		private void _CreateLayerVectors(int[] amountsOfNeurons)
		{
			_layerOutputs = new float[amountsOfNeurons.Length][];

			for (int layer = 0; layer < amountsOfNeurons.Length; layer++)
			{
				_layerOutputs[layer] = new float[amountsOfNeurons[layer]];
			}
		}


		private void _CreateWeightsMatrixes(int[] amountsOfNeurons)
		{
			_weights = new float[amountsOfNeurons.Length - 1][,];

			for (int i = 0; i < _weights.Length; i++)
			{
				int inputLength = amountsOfNeurons[i];
				int outputLength = amountsOfNeurons[i + 1];
				_weights[i] = new float[inputLength, outputLength];
			}
		}


		private void _CreateBiasVectors(int[] amountsOfNeurons)
		{
			_biases = new float[amountsOfNeurons.Length - 1][];

			for (int i = 1; i < amountsOfNeurons.Length; i++)
			{
				_biases[i - 1] = new float[amountsOfNeurons[i]];
			}
		}


		private void _RemoveBiasForNeuron(int layer, int neuron)
		{
			int biasVectorIndex = layer - 1;
			_biases[biasVectorIndex] = _biases[biasVectorIndex].RemoveElementAtIndex(neuron);
		}


		private void _RemoveInWeights(int layer, int neuron)
		{
			int weightMatrixIndex = layer - 1;
			_weights[weightMatrixIndex] = _weights[weightMatrixIndex].RemoveRow(neuron);
		}


		private void _RemoveOutWeights(int layer, int neuron)
		{
			_weights[layer] = _weights[layer].RemoveColumn(neuron);
		}
	}
}
