using MatrixAvxLib;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuroLib;
using NeuroLib.RegularNeuralNetwork.Evolution;
using System;

namespace NeuroLibTest
{
	[TestClass]
	public class RandomParamChangerTest
	{

		/// <summary>
		/// All params (weights and biases) should be changed during the test.
		/// If one of them is eqal to zero, most likely you did something wrong.
		/// </summary>
		[TestMethod]
		public void ItShouldChangeAllParamsInTheNetwork()
		{
			NeuralNetwork net = new (3, 3, 3);
			RandomParamChanger changer = new (new Random());

			for (int i = 0; i < 10000; i++)
			{
				changer.Modify(net);
			}

			TestNetworkParams(net);
		}


		private static void TestNetworkParams(NeuralNetwork net)
		{
			int[] prms = net.GetConstructorParams();
			for (int i = 1; i < prms.Length; i++)
			{
				TestLayerParams(net, i);
			}
		}


		private static void TestLayerParams(NeuralNetwork net, int layer)
		{
			VectorF biases = net.GetBiasVector(layer);
			for (int i = 0; i < biases.Length; i++)
			{
				Assert.AreNotEqual(0, biases[i]);
			}


			MatrixAvxF weights = net.GetWeightMatrix(layer);
			for (int i = 0; i < weights.Width; i++)
			{
				for (int j = 0; j < weights.Height; j++)
				{
					Assert.AreNotEqual(0, weights[i, j]);
				}
			}
		}
	}
}
