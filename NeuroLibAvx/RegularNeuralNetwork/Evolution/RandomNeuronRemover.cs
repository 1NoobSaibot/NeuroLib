using System;
using System.Collections.Generic;

namespace NeuroLib.RegularNeuralNetwork.Evolution
{
	public class RandomNeuronRemover : Modifier<NeuralNetwork>
	{
		private Random _rnd;


		public RandomNeuronRemover() : this(new Random())
		{ }


		public RandomNeuronRemover(Random rnd)
		{
			_rnd = rnd;
		}


		public override NeuralNetwork Modify(NeuralNetwork original)
		{
			_RemoveHiddenNeuron(original);
			return original;
		}


		private void _RemoveHiddenNeuron(NeuralNetwork net)
		{
			int layer = _ChooseLayerToReduce(net);
			int[] prs = net.GetConstructorParams();
			int neuron = _rnd.Next(prs[layer]);
			net.RemoveNeuron(layer, neuron);
		}


		private int _ChooseLayerToReduce(NeuralNetwork net)
		{
			int[] layers = net.GetConstructorParams();
			int[] layersCanBeReduced = _GetLayersCanBeReduced(layers);
			if (layersCanBeReduced.Length == 0)
			{
				throw new CantModifyException("Cannot remove any neuron");
			}
			return layersCanBeReduced[_rnd.Next(layersCanBeReduced.Length)];
		}


		private int[] _GetLayersCanBeReduced(int[] layers)
		{
			List<int> layersCanBeReduced = new List<int>();
			for (int i = 1; i < layers.Length - 1; i++)
			{
				if (layers[i] > 1)
				{
					layersCanBeReduced.Add(i);
				}
			}

			return layersCanBeReduced.ToArray();
		}
	}
}
