namespace NeuroLib
{
	public class ModificationQueue<T> : IModifier<T>
	{
		private IModifier<T>[] _modifiers;


		public ModificationQueue(params IModifier<T>[] modifiers)
		{
			_modifiers = modifiers;
		}


		public T Modify(T original)
		{
			T res = original;
			foreach (var modifier in _modifiers)
			{
				res = modifier.Modify(res);
			}
			return res;
		}
	}
}
