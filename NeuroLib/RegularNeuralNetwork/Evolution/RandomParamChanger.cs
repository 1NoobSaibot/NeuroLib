using MatrixAvxLib;

namespace NeuroLib.RegularNeuralNetwork.Evolution
{
	public class RandomParamChanger : Modifier<NeuralNetwork>
	{
		private readonly Random _rnd;


		public RandomParamChanger(Random rnd)
		{
			_rnd = rnd;
		}


		public override NeuralNetwork Modify(NeuralNetwork original)
		{
			if (_rnd.Next() % 2 == 0)
			{
				ChangeRandomBias(original);
			}
			else
			{
				ChangeRandomWeight(original);
			}

			return original;
		}


		private void ChangeRandomBias(NeuralNetwork net)
		{
			int layer = _rnd.Next(net.AmountOfLayers - 1) + 1;

			VectorF biases = net.GetBiasVector(layer);
			int x = _rnd.Next(biases.Length);

			biases[x] = ChangeValueRandomly(biases[x]);
		}


		private void ChangeRandomWeight(NeuralNetwork net)
		{
			int layer = _rnd.Next(net.AmountOfLayers - 1) + 1;

			MatrixAvxF weights = net.GetWeightMatrix(layer);
			int x = _rnd.Next(weights.Width);
			int y = _rnd.Next(weights.Height);

			weights[x, y] = ChangeValueRandomly(weights[x, y]);
		}


		private float ChangeValueRandomly(float originalValue)
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
