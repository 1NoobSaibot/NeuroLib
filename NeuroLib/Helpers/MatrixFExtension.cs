using MatrixAvxLib;

namespace NeuroLib.Helpers
{
	internal static class MatrixFExtension
	{
		internal static MatrixF RemoveRow(this MatrixF origin, int yIndex)
		{
			MatrixF res = new MatrixF(origin.Width, origin.Height - 1);

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


		internal static MatrixF RemoveColumn(this MatrixF origin, int xIndex)
		{
			MatrixF res = new MatrixF(origin.Width - 1, origin.Height);

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


		internal static MatrixF AddRow(this MatrixF origin)
		{
			MatrixF res = new MatrixF(origin.Width, origin.Height + 1);

			for (int x = 0; x < origin.Width; x++)
			{
				for (int y = 0; y < origin.Height; y++)
				{
					res[x, y] = origin[x, y];
				}
			}

			return res;
		}


		internal static MatrixF AddColumn(this MatrixF origin)
		{
			MatrixF res = new MatrixF(origin.Width + 1, origin.Height);

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
