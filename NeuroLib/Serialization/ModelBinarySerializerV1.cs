using BinaryReaderWriterExtension;

namespace NeuroLib.Serialization
{
	public class ModelBinarySerializerV1 : IModelBinarySerializer
	{
		private const int VERSION = 1;
		public int Version => ModelBinarySerializerV1.VERSION;


		public NeuralNetworkCUDA Deserialize(BinaryReader reader)
		{
			int storedVersion = reader.ReadInt32();
			if (storedVersion != this.Version)
			{
				throw new Exception($"Version mismathc: expected version to be {this.Version}, but version is {storedVersion}");
			}

			int[] hyperParams = reader.ReadInt32Array();
			NeuralNetworkCUDA nn = new(1, hyperParams);

			for (int i = 1; i < hyperParams.Length; i++)
			{
				float[] weights = reader.ReadSingleArray();
				nn.SetWeightsOfLayer(i, weights);
			}

			return nn;
		}


		public void Serialize(NeuralNetworkCUDA nn, BinaryWriter writer)
		{
			writer.Write(Version);

			int[] hyperParams = new int[nn.AmountOfLayers];
			for (int i = 0; i < hyperParams.Length; i++)
			{
				hyperParams[i] = nn.GetNumberOfUnitsForLayer(i);
			}
			writer.WriteArray(hyperParams);

			for (int i = 1; i < hyperParams.Length; i++)
			{
				float[] weights = nn.GetWeightsOfLayer(i);
				writer.WriteArray(weights);
			}
		}
	}
}
