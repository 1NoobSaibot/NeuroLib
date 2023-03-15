using NeuroLib;

namespace NeuroLibTest
{
	[TestClass]
	public class NeuralNetworkTest
	{
		[TestMethod]
		public void ShouldShowAmountOfLayers()
		{
			NeuralNetwork neuralNetwork = new(1, 1, 1);
			Assert.AreEqual(3, neuralNetwork.AmountOfLayers);

			neuralNetwork = new(1, 1, 1, 1);
			Assert.AreEqual(4, neuralNetwork.AmountOfLayers);
		}


		[TestMethod]
		public void ImitatingXor()
		{
			NeuralNetwork xor = XorNetworkBuilder.BuildNetwork();
			XorNetworkTester.TestNetwork(xor);
		}


		[TestMethod]
		public void ClonedNeuralNetworkWorksInTheSameWay()
		{
			NeuralNetwork n1 = XorNetworkBuilder.BuildNetwork();
			NeuralNetwork n2 = n1.Clone();

			XorNetworkTester.TestNetwork(n2);
		}


		[TestMethod]
		public void RemovingOneEmptyNeuronDoesntBreakNetwork()
		{
			NeuralNetwork n1 = XorNetworkBuilder.BuildNetworkWithOverweightHiddenLayer();
			n1.RemoveNeuron(1, 2);
			XorNetworkTester.TestNetwork(n1);

			n1 = XorNetworkBuilder.BuildNetworkWithOverweightInputLayer();
			n1.RemoveNeuron(0, 2);
			XorNetworkTester.TestNetwork(n1);

			n1 = XorNetworkBuilder.BuildNetworkWithOverweightOutputLayer();
			n1.RemoveNeuron(2, 1);
			XorNetworkTester.TestNetwork(n1);
		}


		[TestMethod]
		public void InsertingOneEmptyNeuronDoesntBreakNetwork()
		{
			NeuralNetwork n1 = XorNetworkBuilder.BuildNetwork();
			n1.AddNeuron(1);
			XorNetworkTester.TestNetwork(n1);
			n1.AddNeuron(0);
			XorNetworkTester.TestNetwork(n1);
			n1.AddNeuron(2);
			XorNetworkTester.TestNetwork(n1);
		}


		[TestMethod]
		public void ShouldReturnAnAmountOfParams()
		{
			NeuralNetwork net = new(1, 1);
			Assert.AreEqual(2, net.GetAmountOfParams());

			// (4w + 2b) + (2w + 1b)
			net = new(2, 2, 1);
			Assert.AreEqual(9, net.GetAmountOfParams());
		}
	}
}
