namespace NeuroLib.Helpers
{
	internal static class ArrayExtension
	{
		internal static T[] RemoveElementAtIndex<T>(this T[] origin, int index)
		{
			T[] res = new T[origin.Length - 1];

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


		internal static T[,] RemoveRow<T>(this T[,] origin, int yIndex)
		{
			T[,] res = new T[origin.GetLength(0), origin.GetLength(1) - 1];

			for (int y = 0; y < origin.GetLength(1); y++)
			{
				if (y < yIndex)
				{
					for (int x = 0; x < origin.GetLength(0); x++)
					{
						res[x, y] = origin[x, y];
					}
				}
				else if (y > yIndex)
				{
					for (int x = 0; x < origin.GetLength(0); x++)
					{
						res[x, y - 1] = origin[x, y];
					}
				}
			}

			return res;
		}


		internal static T[,] RemoveColumn<T>(this T[,] origin, int xIndex)
		{
			T[,] res = new T[origin.GetLength(0) - 1, origin.GetLength(1)];

			for (int x = 0; x < origin.GetLength(0); x++)
			{
				if (x < xIndex)
				{
					for (int y = 0; y < origin.GetLength(1); y++)
					{
						res[x, y] = origin[x, y];
					}
				}
				else if (x > xIndex)
				{
					for (int y = 0; y < origin.GetLength(1); y++)
					{
						res[x - 1, y] = origin[x, y];
					}
				}
			}

			return res;
		}
	}
}
