using NeuroLib.Math;

namespace NeuroLib.Layers
{
	public class InputLayer : ILayer
	{
		public readonly CudaDeviceMatrix Inputs;
		private ILayer? NextLayer;
		public CudaDeviceMatrix Outputs => Inputs;


		public InputLayer(int amountOfInputs, int batchSize)
		{
			Inputs = new(rows: batchSize, columns: amountOfInputs);
		}


		public void ApplyNext(ILayer nextLayer)
		{
			if (NextLayer != null)
			{
				throw new ArgumentException("A next layer is already aplied");
			}
			NextLayer = nextLayer;
		}


		public void Forward()
		{
			NextLayer?.Forward();
		}
	}
}
