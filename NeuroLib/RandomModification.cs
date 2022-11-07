﻿using System;
using System.Linq;

namespace NeuroLib
{
	public class RandomModification<T> : Modifier<T>
	{
		private readonly Random _rnd;
		private readonly ModificationWeight<T>[] _modifiers;
		private readonly float _sumOfWeights;


		public RandomModification(ModificationWeight<T>[] modifiers, Random rnd)
		{
			_modifiers = modifiers;
			_rnd = rnd;

			float sumOfWeights = 0;
			for (int i = 0; i < modifiers.Length; i++)
			{
				sumOfWeights += modifiers[i].Weight;
			}
			_sumOfWeights = sumOfWeights;
		}


		public override T Modify(T original)
		{
			Modifier<T> modifier = _ChooseModifier();
			return modifier.Modify(original);
		}


		private Modifier<T> _ChooseModifier()
		{
			float randValue = (float)_rnd.NextDouble() * _sumOfWeights;
			float level = 0;
			for (int i = 0; i < _modifiers.Length - 1; i++)
			{
				level += _modifiers[i].Weight;
				if (level >= randValue)
				{
					return _modifiers[i].Modifier;
				}
			}

			return _modifiers.Last().Modifier;
		}
	}


	public struct ModificationWeight<T>
	{
		public readonly Modifier<T> Modifier;
		public readonly float Weight;


		public ModificationWeight(Modifier<T> modifier, float weight = 1)
		{
			Modifier = modifier;
			Weight = weight;
		}
	}
}
