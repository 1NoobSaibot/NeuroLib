namespace NeuroLib
{
	public class ModificationQueue<T> : Modifier<T>
	{
		private Modifier<T>[] _modifiers;


		public ModificationQueue(params Modifier<T>[] modifiers)
		{
			_modifiers = modifiers;
		}


		public override T Modify(T srcNetwork)
		{
			for (int i = 0; i < _modifiers.Length; i++)
			{
				srcNetwork = _modifiers[i].Modify(srcNetwork);
			}

			return srcNetwork;
		}
	}
}
