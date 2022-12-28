namespace NeuroLib
{
	public interface IModifier<T>
	{
		T Modify(T original);
	}
}
