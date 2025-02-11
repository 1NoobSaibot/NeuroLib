using NeuroLib.Layers;
using NeuroLib.Math;

namespace NeuroLibTest.Networks.Perceptron
{
	[TestClass]
	public class SimplePerceptronTest
	{
		[TestMethod]
		public void CanLearnToBeXOR()
		{
			const float TARGET_MAX_ERROR = 0.01f;
			const int BATCH_SIZE = 1;
			const int seed = 184131478;

			Random rnd = new Random(seed);
			InputLayer input = new(2, BATCH_SIZE);
			DenseLayerTrainable hiddenLayer = new(input, 2);
			hiddenLayer.RandomiseWeights(rnd);
			DenseLayerTrainable output = new(hiddenLayer, 1);
			output.RandomiseWeights(rnd);

			TrainableLayerTracker hiddenTracker = new(hiddenLayer);
			TrainableLayerTracker outputTracker = new(output);

			float[][] inputDataSets = [
				[ 0, 0 ],
				[ 0, 1 ],
				[ 1, 0 ],
				[ 1, 1 ],
			];

			float[][] trueOutValues = [[0], [1], [1], [0]];

			float error = Fit();

			Assert.IsTrue(error <= TARGET_MAX_ERROR, $"The error={error} is bigger than expected: {TARGET_MAX_ERROR}");
			Console.WriteLine("The error is just " + error);

			float Fit()
			{
				int[] dataSetOrder = [0, 1, 2, 3];
				CudaDeviceMatrix trueOut = new(rows: 1, columns: 1);
				float[] actualVectorBuffer = new float[1];
				float lastError = float.MaxValue;
				for (int epoch = 0; epoch < 1000; epoch++)
				{
					rnd.Shuffle(dataSetOrder);
					float theBiggestErrorInTheEpoch = 0;
					foreach (var i in dataSetOrder)
					{
						float[] inputVector = inputDataSets[i];
						float[] trueOutputVector = trueOutValues[i];

						input.Inputs.SetValues(inputVector);
						input.Forward();
						output.Outputs.GetValues(actualVectorBuffer);
						float error = GetMaxAbsoluteError(trueOutputVector, actualVectorBuffer);
						if (error > theBiggestErrorInTheEpoch)
						{
							theBiggestErrorInTheEpoch = error;
						}

						trueOut.SetValues(trueOutputVector);
						output.StartBackward(trueOut);
					}
					lastError = theBiggestErrorInTheEpoch;
				}

				return lastError;
			}

			void TrackDeviceData()
			{
				hiddenTracker.Update();
				outputTracker.Update();
			}
		}


		private static float GetMaxAbsoluteError(float[] expected, float[] actual)
		{
			Assert.AreEqual(expected.Length, actual.Length);

			float maxAbsoluteError = 0;
			for (int i = 0; i < expected.Length; i++)
			{
				float localError = Math.Abs(expected[i] - actual[i]);
				if (localError > maxAbsoluteError)
				{
					maxAbsoluteError = localError;
				}
			}

			return maxAbsoluteError;
		}


		private class TrainableLayerTracker
		{
			public float[] Inputs;
			public float[] Weights;
			public float[] ZSum;
			public float[] Outputs;
			public float[] InputDeltas;
			public float[] OutputDeltas;
			private readonly DenseLayerTrainable _m;


			public TrainableLayerTracker(DenseLayerTrainable layer)
			{
				_m = layer;

				Inputs = new float[layer.Inputs.ElementCount];
				Weights = new float[layer.WeightsAndBiases.ElementCount];
				ZSum = new float[layer.Sum_Z.ElementCount];
				Outputs = new float[layer.Outputs.ElementCount];
				InputDeltas = new float[layer.InputDeltas.ElementCount];
				OutputDeltas = new float[layer.OutputDeltas.ElementCount];
			}


			public void Update()
			{
				_m.Inputs.GetValues(Inputs);
				_m.WeightsAndBiases.GetValues(Weights);
				_m.Sum_Z.GetValues(ZSum);
				_m.Outputs.GetValues(Outputs);
				_m.InputDeltas.GetValues(InputDeltas);
				_m.OutputDeltas.GetValues(OutputDeltas);
			}
		}
	}
}
