using System;

namespace NeuroLib
{
	public class CantModifyException : Exception
	{
		public CantModifyException(string message, object original)
			: base(message)
		{
			Data.Add("Original", original);
		}


		public CantModifyException(string message, object original, CantModifyException[] errors)
			: this(message, original)
		{
			Data.Add("InnerErrors", errors);
		}
	}
}
