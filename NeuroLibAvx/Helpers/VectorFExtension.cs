using MatrixAvxLib;

namespace NeuroLib.Helpers
{
	internal static class VectorFExtension
	{
		internal static VectorF RemoveElementAtIndex(this VectorF origin, int index)
		{
			VectorF res = new VectorF(origin.Length - 1);

			for (int i = 0; i < origin.Length; i++)
			{
				if (i < index)
				{
					res[i] = origin[i];
				}
				else if (i > index)
				{
					res[i - 1] = origin[i];
				}
			}

			return res;
		}


		internal static VectorF Extend(this VectorF origin)
		{
			VectorF res = new VectorF(origin.Length + 1);

			for (int i = 0; i < origin.Length; i++)
			{
				res[i] = origin[i];
			}

			return res;
		}
	}
}
