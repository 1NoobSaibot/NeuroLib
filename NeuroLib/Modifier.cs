namespace NeuroLib
{
	public abstract class Modifier<T> : IModifier<T>
	{
		public abstract T Modify(T original);
	}
}
