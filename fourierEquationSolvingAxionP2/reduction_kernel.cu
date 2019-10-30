#include "reduction.h"

namespace cg = cooperative_groups;

__device__ __forceinline__ double getMax(double x, double y) {
	return ((x > y) ? x : y);
}

__device__ __forceinline__ double getSum(double x, double y) {
	return x + y;
}

__global__ void reduceKernel(double (*f)(double x, double y), double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	cta.sync();
	

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = getSum(sdata[tid], sdata[tid + s]);
		}

		cta.sync();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


void reduce(int type, int size, int threads, int blocks, double *d_idata, double *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = threads * sizeof(double);

	switch (type)
	{
	case MAXIMUM:
		reduceKernel << <dimGrid, dimBlock, smemSize >> > (getMax, d_idata, d_odata, size);
		break;
	case SUMMATION:
		reduceKernel << <dimGrid, dimBlock, smemSize >> > (getSum, d_idata, d_odata, size);
		break;
	case MEAN:
		reduceKernel << <dimGrid, dimBlock, smemSize >> > (getSum, d_idata, d_odata, size);
		break;
	case SIGMA2:
		reduceKernel <<<dimGrid, dimBlock, smemSize>>> (getSum, d_idata, d_odata, size);
		break;
	case SIGMA4:
		reduceKernel <<<dimGrid, dimBlock, smemSize>>> (getSum, d_idata, d_odata, size);
		break;


	default:
		break;
	}
}



