using NeuroLib.Math;

namespace NeuroLib.Layers
{
	public interface ILayer
	{
		CudaDeviceMatrix Outputs { get; }
		void ApplyNext(ILayer nextLayer);
		void Forward();
	}
}
