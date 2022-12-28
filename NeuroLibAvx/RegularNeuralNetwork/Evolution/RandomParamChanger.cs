using MatrixAvxLib;
using System;

namespace NeuroLib.RegularNeuralNetwork.Evolution
{
	public class RandomParamChanger : Modifier<NeuralNetwork>
	{
		private Random _rnd;


		public RandomParamChanger(Random rnd)
		{
			_rnd = rnd;
		}


		public override NeuralNetwork Modify(NeuralNetwork original)
		{
			if (_rnd.Next() % 2 == 0)
			{
				_ChangeRandomBias(original);
			}
			else
			{
				_ChangeRandomWeight(original);
			}

			return original;
		}


		private void _ChangeRandomBias(NeuralNetwork net)
		{
			int layer = _rnd.Next(net.AmountOfLayers - 1) + 1;

			float[] biases = net.GetBiasVector(layer);
			int x = _rnd.Next(biases.Length);

			biases[x] = _ChangeValueRandomly(biases[x]);
		}


		private void _ChangeRandomWeight(NeuralNetwork net)
		{
			int layer = _rnd.Next(net.AmountOfLayers - 1) + 1;

			MatrixF weights = net.GetWeightMatrix(layer);
			int x = _rnd.Next(weights.Width);
			int y = _rnd.Next(weights.Height);

			weights[x, y] = _ChangeValueRandomly(weights[x, y]);
		}


		private float _ChangeValueRandomly(float originalValue)
		{
			int precision = _rnd.Next(7);
			int order = originalValue == 0.0f
			? 0
				: (int)Math.Log10(Math.Abs(originalValue));

			float scale = (float)Math.Pow(10, order - precision);
			float difference = (float)_rnd.NextDouble() * 2 - 1;
			return originalValue + scale * difference;
		}
	}
}
