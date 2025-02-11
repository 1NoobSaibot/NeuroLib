using NeuroLib.Layers;

namespace NeuroLibTest.Layers
{
	[TestClass]
	public class DenseLayerTest
	{
		private const int inputVectorSize = 2;
		private const int outputVectorSize = 3;
		private const int batchSize = 5;

		private readonly float[] inputBuffer = new float[inputVectorSize * batchSize];
		private readonly float[] weightsBuffer = new float[(inputVectorSize + 1) * outputVectorSize];
		private readonly float[] outputBuffer = new float[outputVectorSize * batchSize];


		[TestMethod]
		public void ReturnZerosWhenNotInitialized()
		{
			InputLayer input = new(amountOfInputs: inputVectorSize, batchSize: batchSize);
			DenseLayer layer = new(input, outputVectorSize);

			layer.Forward();
			layer.Outputs.GetValues(outputBuffer);

			for (int i = 0; i < outputBuffer.Length; i++)
			{
				Assert.AreEqual(0, outputBuffer[i]);
			}
		}


		[TestMethod]
		public void InputWeightOutputAreBoundWell()
		{
			InputLayer input = new(amountOfInputs: inputVectorSize, batchSize: batchSize);
			DenseLayer layer = new(input, outputVectorSize);

			Reset(inputBuffer);
			Reset(weightsBuffer);

			for (int i = 0; i < inputVectorSize; i++)
			{
				for (int ib = 0; ib < batchSize; ib++)
				{
					inputBuffer[inputVectorSize * ib + i] = 1;
					layer.Inputs.SetValues(inputBuffer);

					for (int o = 0; o < outputVectorSize; o++)
					{
						for (int wi = 0; wi < inputVectorSize; wi++)
						{
							weightsBuffer[outputVectorSize * wi + o] = 1;
							layer.WeightsAndBiases.SetValues(weightsBuffer);

							layer.Forward();
							layer.Outputs.GetValues(outputBuffer);

							bool expectedAnOutputToBeSet = i == wi;
							int expectedSetIndex = -1;
							if (expectedAnOutputToBeSet)
							{
								expectedSetIndex = outputVectorSize * ib + o;
							}

							for (int oi = 0; oi < outputBuffer.Length; oi++)
							{
								if (oi == expectedSetIndex)
								{
									Assert.AreEqual(1, outputBuffer[oi]);
								}
								else
								{
									Assert.AreEqual(0, outputBuffer[oi]);
								}
							}

							weightsBuffer[outputVectorSize * wi + o] = 0;
						}
					}

					inputBuffer[inputVectorSize * ib + i] = 0;
				}
			}
		}


		[TestMethod]
		public void BiasesOutputsAreBound()
		{
			InputLayer input = new(amountOfInputs: inputVectorSize, batchSize: batchSize);
			DenseLayer layer = new(input, outputVectorSize);

			Reset(inputBuffer);
			Reset(weightsBuffer);

			int biasWeightIndex = inputVectorSize;
			for (int o = 0; o < outputVectorSize; o++)
			{
				weightsBuffer[outputVectorSize * biasWeightIndex + o] = 1;
				layer.WeightsAndBiases.SetValues(weightsBuffer);

				layer.Forward();
				layer.Outputs.GetValues(outputBuffer);

				int expectedOutputsSat = o;

				for (int oi = 0; oi < outputBuffer.Length; oi++)
				{
					int batchIndex = oi / outputVectorSize;
					int outputIndex = oi % outputVectorSize;
					if (outputIndex == expectedOutputsSat)
					{
						Assert.AreEqual(1, outputBuffer[oi]);
					}
					else
					{
						Assert.AreEqual(0, outputBuffer[oi]);
					}
				}

				weightsBuffer[outputVectorSize * biasWeightIndex + o] = 0;
			}
		}


		private static void Reset(float[] array)
		{
			for (int i = 0; i < array.Length; i++)
			{
				array[i] = 0;
			}
		}
	}
}
