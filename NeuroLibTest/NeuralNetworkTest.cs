using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuroLib;
using System;

namespace NeuroLibTest
{
	[TestClass]
	public class NeuralNetworkTest
	{
		[TestMethod]
		public void ShouldWorkWellWithoutTrowingAnyErrors()
		{
			NeuralNetwork neuralNetwork = new NeuralNetwork(4, 3, 2);
			neuralNetwork.Tick();
		}


		[TestMethod]
		public void ImitatingXor()
		{
			NeuralNetwork xor = new NeuralNetwork(2, 2, 1);

			xor.GetWeightMatrix(1)[0, 0] = -1.5f;
			xor.GetWeightMatrix(1)[0, 1] = +1.4f;
			xor.GetWeightMatrix(1)[1, 0] = +1.4f;
			xor.GetWeightMatrix(1)[1, 1] = -1.5f;

			xor.GetWeightMatrix(2)[0, 0] = +2.5f;
			xor.GetWeightMatrix(2)[1, 0] = +2.5f;

			xor.GetBiasVector(1)[0] = -0.8f;
			xor.GetBiasVector(1)[1] = -0.8f;
			xor.GetBiasVector(2)[0] = -1.25f;

			xor.SetInputs(new float[] { 0, 0 });
			xor.Tick();
			Assert.IsTrue(xor.GetOutput(0) < 0.1);

			xor.SetInputs(new float[] { 0, 1 });
			xor.Tick();
			Assert.IsTrue(xor.GetOutput(0) > 0.9);

			xor.SetInputs(new float[] { 1, 0 });
			xor.Tick();
			Assert.IsTrue(xor.GetOutput(0) > 0.9);

			xor.SetInputs(new float[] { 1, 1 });
			xor.Tick();
			Assert.IsTrue(xor.GetOutput(0) < 0.1);
		}
	}
}
