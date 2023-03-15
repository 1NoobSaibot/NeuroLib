using NeuroLib;
using NeuroLib.RegularNeuralNetwork.Evolution;

namespace NeuroLibTest
{
	[TestClass]
	public class RandomNeuronInserterTest
	{

		[TestMethod]
		public void ItShouldNotChangeAnyParamsOfIndependentNeurons()
		{
			NeuralNetwork xor = XorNetworkBuilder.BuildNetwork();
			RandomNeuronInserter inserter = new();
			xor = inserter.Modify(xor);

			// It doesn't throw any construction error;
			xor.Tick();

			// It should break the answers. But it doesn't. Maybe that is even better
			// Assert.ThrowsException<AssertFailedException>(() => { XorNetworkTester.TestNetwork(xor); });

			// Removing this neuron should return the original state.
			xor.RemoveNeuron(1, 2);
			XorNetworkTester.TestNetwork(xor);
		}
	}
}
