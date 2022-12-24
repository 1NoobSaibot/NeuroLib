using System;

namespace NeuroLib
{
	public class CantModifyException : Exception
	{
		public CantModifyException(string message) : base(message)
		{
		}


		public CantModifyException(string message, CantModifyException[] errors)
			: this(message)
		{
			Data.Add("InnerErrors", errors);
		}
	}
}
