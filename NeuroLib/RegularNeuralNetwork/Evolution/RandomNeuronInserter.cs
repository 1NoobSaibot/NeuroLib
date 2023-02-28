using MatrixAvxLib;

namespace NeuroLib.RegularNeuralNetwork.Evolution
{
	public class RandomNeuronInserter : Modifier<NeuralNetwork>
	{
		private readonly Random _rnd;


		public RandomNeuronInserter() : this(new Random())
		{ }


		public RandomNeuronInserter(Random rnd)
		{
			_rnd = rnd;
		}


		public override NeuralNetwork Modify(NeuralNetwork original)
		{
			AddHiddenNeuron(original);
			return original;
		}


		private void AddHiddenNeuron(NeuralNetwork net)
		{
			int[] prs = net.GetConstructorParams();
			if (prs.Length <= 2)
			{
				throw new CantModifyException("There is no hidden layers to insert neuron");
			}

			int layer = ChooseLayerToExtend(prs);
			net.AddNeuron(layer);

			RandomiseNewParams(net, layer);
		}


		private int ChooseLayerToExtend(int[] layers)
		{
			return _rnd.Next(layers.Length - 2) + 1;
		}


		private void RandomiseNewParams(NeuralNetwork net, int layer)
		{
			RandomiseBias(net, layer);
			RandomiseInWeights(net, layer);
			CreateOneRandomBindWithNextLayer(net, layer);
		}


		private void RandomiseBias(NeuralNetwork net, int atLayer)
		{
			VectorF biasVector = net.GetBiasVector(atLayer);
			biasVector[biasVector.Length - 1] = ((float)_rnd.NextDouble() * 2) - 1;
		}


		private void RandomiseInWeights(NeuralNetwork net, int layer)
		{
			MatrixAvxF inWeights = net.GetWeightMatrix(layer);
			int insertedRowIndex = inWeights.Height - 1;
			for (int i = 0; i < inWeights.Width; i++)
			{
				inWeights[i, insertedRowIndex] = ((float)_rnd.NextDouble() * 2) - 1;
			}
		}


		private void CreateOneRandomBindWithNextLayer(NeuralNetwork net, int layer)
		{
			MatrixAvxF outWeights = net.GetWeightMatrix(layer + 1);
			int insertedColumnIndex = outWeights.Width - 1;
			int randomNextNeuron = _rnd.Next(outWeights.Height);
			outWeights[insertedColumnIndex, randomNextNeuron] = ((float)_rnd.NextDouble() * 2) - 1;
		}
	}
}
