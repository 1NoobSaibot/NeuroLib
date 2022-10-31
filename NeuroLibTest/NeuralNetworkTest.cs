using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuroLib;

namespace NeuroLibTest
{
	[TestClass]
	public class NeuralNetworkTest
	{
		[TestMethod]
		public void ShouldShowAmountOfLayers()
		{
			NeuralNetwork neuralNetwork = new NeuralNetwork(1, 1, 1);
			Assert.AreEqual(3, neuralNetwork.AmountOfLayers);

			neuralNetwork = new NeuralNetwork(1, 1, 1, 1);
			Assert.AreEqual(4, neuralNetwork.AmountOfLayers);
		}


		[TestMethod]
		public void ShouldWorkWellWithoutTrowingAnyErrors()
		{
			NeuralNetwork neuralNetwork = new NeuralNetwork(4, 3, 2);
			neuralNetwork.Tick();
		}


		[TestMethod]
		public void ImitatingXor()
		{
			NeuralNetwork xor = XorNetworkBuilder.BuildNetwork();
			XorNetworkTester.TestNetwork(xor);
		}
	}
}
