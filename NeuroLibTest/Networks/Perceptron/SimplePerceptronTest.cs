using NeuroLib;

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

			Random rnd = new(seed);
			NeuralNetworkCUDA nn = new(BATCH_SIZE, [2, 2, 1]);
			nn.Randomize(rnd);

			float[][] inputDataSets = [
				[ 0, 0 ],
				[ 0, 1 ],
				[ 1, 0 ],
				[ 1, 1 ],
			];

			float[][] trueOutValues = [[0], [1], [1], [0]];
			nn.Fit(rnd, inputDataSets, trueOutValues, epochs: 1000);

			float maxAbsError = 0;
			float[] outputBuffer = new float[1];
			for (int i = 0; i < inputDataSets.Length; i++)
			{
				nn.Forward(inputDataSets[i], outputBuffer);
				float localError = Math.Abs(outputBuffer[0] - trueOutValues[i][0]);
				if (localError > maxAbsError)
				{
					maxAbsError = localError;
				}
			}

			Assert.IsTrue(maxAbsError <= TARGET_MAX_ERROR, $"The error={maxAbsError} is bigger than expected: {TARGET_MAX_ERROR}");
			Console.WriteLine("The error is just " + maxAbsError);
		}
	}
}
