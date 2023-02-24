using MatrixAvxLib;

namespace NeuroLib.Helpers
{
	internal static class MatrixAvxFExtension
	{
		internal static MatrixAvxF RemoveRow(this MatrixAvxF origin, int yIndex)
		{
			MatrixAvxF res = new MatrixAvxF(origin.Width, origin.Height - 1);

			for (int y = 0; y < origin.Height; y++)
			{
				if (y < yIndex)
				{
					for (int x = 0; x < origin.Width; x++)
					{
						res[x, y] = origin[x, y];
					}
				}
				else if (y > yIndex)
				{
					for (int x = 0; x < origin.Width; x++)
					{
						res[x, y - 1] = origin[x, y];
					}
				}
			}

			return res;
		}


		internal static MatrixAvxF RemoveColumn(this MatrixAvxF origin, int xIndex)
		{
			MatrixAvxF res = new MatrixAvxF(origin.Width - 1, origin.Height);

			for (int x = 0; x < origin.Width; x++)
			{
				if (x < xIndex)
				{
					for (int y = 0; y < origin.Height; y++)
					{
						res[x, y] = origin[x, y];
					}
				}
				else if (x > xIndex)
				{
					for (int y = 0; y < origin.Height; y++)
					{
						res[x - 1, y] = origin[x, y];
					}
				}
			}

			return res;
		}


		internal static MatrixAvxF AddRow(this MatrixAvxF origin)
		{
			MatrixAvxF res = new MatrixAvxF(origin.Width, origin.Height + 1);

			for (int x = 0; x < origin.Width; x++)
			{
				for (int y = 0; y < origin.Height; y++)
				{
					res[x, y] = origin[x, y];
				}
			}

			return res;
		}


		internal static MatrixAvxF AddColumn(this MatrixAvxF origin)
		{
			MatrixAvxF res = new MatrixAvxF(origin.Width + 1, origin.Height);

			for (int x = 0; x < origin.Width; x++)
			{
				for (int y = 0; y < origin.Height; y++)
				{
					res[x, y] = origin[x, y];
				}
			}

			return res;
		}
	}
}
