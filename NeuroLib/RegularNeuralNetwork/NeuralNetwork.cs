using MatrixAvxLib;
using NeuroLib.Helpers;
using System.Diagnostics;


namespace NeuroLib
{
	public class NeuralNetwork
	{
		private MatrixF[] _weights;
		private VectorF[] _layerOutputs;
		private VectorF[] _biases;


		public int AmountOfLayers => _layerOutputs.Length;

		
		public NeuralNetwork(params int[] amountsOfNeurons)
		{
			Debug.Assert(amountsOfNeurons.Length >= 2);

			_layerOutputs = _CreateLayerVectors(amountsOfNeurons);
			_weights = _CreateWeightsMatrixes(amountsOfNeurons);
			_biases = _CreateBiasVectors(amountsOfNeurons);
		}


		public void SetInput(int index, float value)
		{
			_layerOutputs[0][index] = value;
		}


		public void SetInputs(float[] values)
		{
			int length = Math.Min(_layerOutputs[0].Length, values.Length);
			for (int i = 0; i < length; i++)
			{
				_layerOutputs[0][i] = values[i];
			}
		}


		public void SetInputs(VectorF values)
		{
			int length = Math.Min(_layerOutputs[0].Length, values.Length);
			for (int i = 0; i < length; i++)
			{
				_layerOutputs[0][i] = values[i];
			}
		}


		public void Tick()
		{
			for (int outLayerIndex = 1; outLayerIndex < _layerOutputs.Length; outLayerIndex++)
			{
				VectorF outLayer = _layerOutputs[outLayerIndex];
				VectorF inLayer = _layerOutputs[outLayerIndex - 1];
				VectorF biasVector = _biases[outLayerIndex - 1];
				MatrixF weightMatrix = _weights[outLayerIndex - 1];

				MatrixMath.Mul(weightMatrix, inLayer, outLayer);
				MatrixMath.Add(outLayer, biasVector, outLayer);

				for (int i = 0; i < outLayer.Length; i++)
				{
					outLayer[i] = _Sigmoid(outLayer[i]);
				}
			}


			float _Sigmoid(float input)
			{
				return 1 / (1 + (float)Math.Exp(-4 * input));
			}
		}


		/// <summary>
		/// Returns count of weights and biases. The less params it has, the fastest calculations.
		/// When you use genetic, you should prioritize the smalest networks between the most accurate.
		/// </summary>
		/// <returns>Count of weights and biases together</returns>
		public int GetAmountOfParams()
		{
			int[] parms = GetConstructorParams();

			int sum = 0;
			for (int i = 1; i < parms.Length; i++)
			{
				sum += parms[i] + parms[i - 1] * parms[i];
			}

			return sum;
		}


		public float GetOutput(int outIndex)
		{
			int outLayerIndex = _layerOutputs.Length - 1;
			return _layerOutputs[outLayerIndex][outIndex];
		}


		public MatrixF GetWeightMatrix(int outLayerIndex)
		{
			return _weights[outLayerIndex - 1];
		}


		public VectorF GetBiasVector(int layerIndex)
		{
			return _biases[layerIndex - 1];
		}


		public NeuralNetwork Clone()
		{
			int[] layers = GetConstructorParams();
			NeuralNetwork copy = new NeuralNetwork(layers);

			for (int i = 1; i < layers.Length; i++)
			{
				VectorF dstBiasVector = copy.GetBiasVector(i);
				VectorF srcBiasVector = this.GetBiasVector(i);
				MatrixF dstWeightMatrix = copy.GetWeightMatrix(i);
				MatrixF srcWeightMatrix = this.GetWeightMatrix(i);

				for (int j = 0; j < dstBiasVector.Length; j++)
				{
					dstBiasVector[j] = srcBiasVector[j];

					for (int inNeuron = 0; inNeuron < dstWeightMatrix.Width; inNeuron++)
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


		public void AddNeuron(int layer)
		{
			if (layer == 0)
			{
				AddInputNeuron();
				return;
			}
			if (layer == _layerOutputs.Length - 1)
			{
				AddOutputNeuron();
				return;
			}

			_layerOutputs[layer] = _layerOutputs[layer].Extend();
			_biases[layer - 1] = _biases[layer - 1].Extend();
			_weights[layer - 1] = _weights[layer - 1].AddRow();
			_weights[layer] = _weights[layer].AddColumn();
		}


		public void AddInputNeuron()
		{
			_layerOutputs[0] = _layerOutputs[0].Extend();
			_weights[0] = _weights[0].AddColumn();
		}


		public void AddOutputNeuron()
		{
			int layer = _layerOutputs.Length - 1;
			_layerOutputs[layer] = _layerOutputs[layer].Extend();
			_biases[layer - 1] = _biases[layer - 1].Extend();
			_weights[layer - 1] = _weights[layer - 1].AddRow();
		}


		private VectorF[] _CreateLayerVectors(int[] amountsOfNeurons)
		{
			var layers = new VectorF[amountsOfNeurons.Length];

			for (int layer = 0; layer < amountsOfNeurons.Length; layer++)
			{
				layers[layer] = new VectorF(amountsOfNeurons[layer]);
			}

			return layers;
		}


		private MatrixF[] _CreateWeightsMatrixes(int[] amountsOfNeurons)
		{
			var weights = new MatrixF[amountsOfNeurons.Length - 1];

			for (int i = 0; i < weights.Length; i++)
			{
				int inputLength = amountsOfNeurons[i];
				int outputLength = amountsOfNeurons[i + 1];
				weights[i] = new MatrixF(inputLength, outputLength);
			}

			return weights;
		}


		private VectorF[] _CreateBiasVectors(int[] amountsOfNeurons)
		{
			var biases = new VectorF[amountsOfNeurons.Length - 1];

			for (int i = 1; i < amountsOfNeurons.Length; i++)
			{
				biases[i - 1] = new VectorF(amountsOfNeurons[i]);
			}

			return biases;
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
