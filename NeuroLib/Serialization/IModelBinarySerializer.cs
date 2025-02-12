namespace NeuroLib.Serialization
{
	internal interface IModelBinarySerializer
	{
		int Version { get; }
		void Serialize(NeuralNetworkCUDA nn, BinaryWriter writer);
		NeuralNetworkCUDA Deserialize(BinaryReader reader);
	}
}
