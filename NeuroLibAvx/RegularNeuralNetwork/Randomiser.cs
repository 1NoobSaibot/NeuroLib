using MatrixAvxLib;

namespace NeuroLib
{
	public class NeuralNetworkRandomiser : Modifier<NeuralNetwork>
	{
		private Random _rnd;

		public NeuralNetworkRandomiser()
		{
			_rnd = new Random();
		}


		public NeuralNetworkRandomiser(Random rand)
		{
			_rnd = rand;
		}


		public override NeuralNetwork Modify(NeuralNetwork srcNetwork)
		{
			int[] layers = srcNetwork.GetConstructorParams();

			for (int i = 1; i < layers.Length; i++)
			{
				VectorF biasVector = srcNetwork.GetBiasVector(i);
				MatrixF weightMatrix = srcNetwork.GetWeightMatrix(i);

				for (int j = 0; j < biasVector.Length; j++)
				{
					biasVector[j] = (float)_rnd.NextDouble() * 2 - 1;

					for (int inNeuron = 0; inNeuron < weightMatrix.Width; inNeuron++)
					{
						weightMatrix[inNeuron, j] = (float)_rnd.NextDouble() * 2 - 1;
					}
				}
			}

			return srcNetwork;
		}
	}
}
