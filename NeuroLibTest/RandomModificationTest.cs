using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuroLib;
using NeuroLib.RegularNeuralNetwork.Evolution;
using System;

namespace NeuroLibTest
{
	[TestClass]
	public class RandomModificationTest
	{

		[TestMethod]
		public void ItShouldCallAllOfModificationsDuringTime()
		{
			RandomModification<TestModel> randModifier = new RandomModification<TestModel>(
				new ModificationWeight<TestModel>[3]
				{
					new ModificationWeight<TestModel>(new TestModification(0), 1),
					new ModificationWeight<TestModel>(new TestModification(1), 1),
					new ModificationWeight<TestModel>(new TestModification(2), 1)
				},
				new Random()
			);

			TestModel model = new TestModel(3);

			for (int i = 0; i < 100; i++)
			{
				randModifier.Modify(model);
			}

			for (int i = 0; i < model.ModificationCounters.Length; i++)
			{
				Assert.AreNotEqual(0, model.ModificationCounters[i]);
			}
		}


		private class TestModification : Modifier<TestModel>
		{
			public readonly int Id;

			public TestModification(int id)
			{
				Id = id;
			}

			public override TestModel Modify(TestModel original)
			{
				original.ModificationCounters[Id]++;
				return original;
			}
		}


		private class TestModel
		{
			public int[] ModificationCounters;

			public TestModel(int length)
			{
				ModificationCounters = new int[length];
			}
		}
	}
}
