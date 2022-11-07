namespace NeuroLib
{
	public class NeuralNetworkCloner : Modifier<NeuralNetwork>
	{
		public override NeuralNetwork Modify(NeuralNetwork srcNetwork)
		{
			return srcNetwork.Clone();
		}
	}
}
