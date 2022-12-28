using NeuroLib;

namespace NeuroLibTest
{
	internal static class XorNetworkBuilder
	{
		public static NeuralNetwork BuildNetwork()
		{
			NeuralNetwork xor = new NeuralNetwork(2, 2, 1);
			_SetUpWeightsAndBiases(xor);
			return xor;
		}


		/// <summary>
		/// Creates a neural network with 3 neurons on layer 2.
		/// Third neuron is useless
		/// </summary>
		/// <returns></returns>
		internal static NeuralNetwork BuildNetworkWithOverweightHiddenLayer()
		{
			NeuralNetwork xor = new NeuralNetwork(2, 3, 1);
			_SetUpWeightsAndBiases(xor);
			return xor;
		}


		internal static NeuralNetwork BuildNetworkWithOverweightInputLayer()
		{
			NeuralNetwork xor = new NeuralNetwork(3, 2, 1);
			_SetUpWeightsAndBiases(xor);
			return xor;
		}


		internal static NeuralNetwork BuildNetworkWithOverweightOutputLayer()
		{
			NeuralNetwork xor = new NeuralNetwork(2, 2, 2);
			_SetUpWeightsAndBiases(xor);
			return xor;
		}


		private static void _SetUpWeightsAndBiases(NeuralNetwork xor)
		{
			xor.GetWeightMatrix(1)[0, 0] = -1.5f;
			xor.GetWeightMatrix(1)[0, 1] = +1.4f;
			xor.GetWeightMatrix(1)[1, 0] = +1.4f;
			xor.GetWeightMatrix(1)[1, 1] = -1.5f;

			xor.GetWeightMatrix(2)[0, 0] = +2.5f;
			xor.GetWeightMatrix(2)[1, 0] = +2.5f;

			xor.GetBiasVector(1)[0] = -0.8f;
			xor.GetBiasVector(1)[1] = -0.8f;
			xor.GetBiasVector(2)[0] = -1.25f;
		}
	}
}
