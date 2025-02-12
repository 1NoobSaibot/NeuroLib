using NeuroLib;
using NeuroLib.Serialization;
using NeuroLibTest.Networks.Perceptron;

namespace NeuroLibTest.Serialization
{
	[TestClass]
	public class ModelBinarySerializerV1Test
	{
		[TestMethod]
		public void CanSerializeDeserializeAndUseDeserializedModelInTheSameWay()
		{
			NeuralNetworkCUDA originalNN = SimplePerceptronTest.CreateXorExample();
			using MemoryStream stream = new();
			using BinaryWriter writer = new(stream);
			ModelBinarySerializerV1 serializer = new();
			serializer.Serialize(originalNN, writer);

			stream.Position = 0;

			using BinaryReader reader = new(stream);
			NeuralNetworkCUDA readNN = serializer.Deserialize(reader);
			SimplePerceptronTest.AssertErrorIsSmallEnough(readNN);
		}
	}
}
