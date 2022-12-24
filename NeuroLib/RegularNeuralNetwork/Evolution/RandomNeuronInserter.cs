using System;

namespace NeuroLib.RegularNeuralNetwork.Evolution
{
	public class RandomNeuronInserter : Modifier<NeuralNetwork>
	{
		private Random _rnd;


		public RandomNeuronInserter() : this(new Random())
		{ }


		public RandomNeuronInserter(Random rnd)
		{
			_rnd = rnd;
		}


		public override NeuralNetwork Modify(NeuralNetwork original)
		{
			_AddHiddenNeuron(original);
			return original;
		}


		private void _AddHiddenNeuron(NeuralNetwork net)
		{
			int[] prs = net.GetConstructorParams();
			if (prs.Length <= 2)
			{
				throw new CantModifyException("There is no hidden layers to insert neuron");
			}

			int layer = _ChooseLayerToExtend(prs);
			net.AddNeuron(layer);

			_RandomiseNewParams(net, layer);
		}


		private int _ChooseLayerToExtend(int[] layers)
		{
			return _rnd.Next(layers.Length - 2) + 1;
		}


		private void _RandomiseNewParams(NeuralNetwork net, int layer)
		{
			_RandomiseBias(net, layer);
			_RandomeseInWeights(net, layer);
			_CreateOneRandomBindWithNextLayer(net, layer);
		}


		private void _RandomiseBias(NeuralNetwork net, int atLayer)
		{
			float[] biasVector = net.GetBiasVector(atLayer);
			biasVector[biasVector.Length - 1] = ((float)_rnd.NextDouble() * 2) - 1;
		}


		private void _RandomeseInWeights(NeuralNetwork net, int layer)
		{
			float[,] inWeights = net.GetWeightMatrix(layer);
			int insertedRowIndex = inWeights.GetLength(1) - 1;
			for (int i = 0; i < inWeights.GetLength(0); i++)
			{
				inWeights[i, insertedRowIndex] = ((float)_rnd.NextDouble() * 2) - 1;
			}
		}


		private void _CreateOneRandomBindWithNextLayer(NeuralNetwork net, int layer)
		{
			float[,] outWeights = net.GetWeightMatrix(layer + 1);
			int insertedColumnIndex = outWeights.GetLength(0) - 1;
			int randomNextNeuron = _rnd.Next(outWeights.GetLength(1));
			outWeights[insertedColumnIndex, randomNextNeuron] = ((float)_rnd.NextDouble() * 2) - 1;
		}
	}
}
