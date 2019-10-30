#include "reduction.h"

void reduce(int type, int size, int threads, int blocks, double *d_idata, double *d_odata);

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

void getNumBlocksAndThreads(int n, int maxThreads, int &blocks, int &threads)
{
	threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
	blocks = (n + threads - 1) / threads;
}

double reductionMax(int size, double *inData)
{
	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);
	
	double *inData_dev = NULL;
	double *outData_dev = NULL;
	
	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));
			
	reduce(MAXIMUM, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(MAXIMUM, s, threads, blocks, inData_dev, outData_dev);
		
		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = outData_host[0];
	for (size_t i = 1; i < s; i++)
	{
		result = result > outData_host[i] ? result : outData_host[i];
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

double reductionSum(int size, double *inData)
{

	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(SUMMATION, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(SUMMATION, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result += outData_host[i];
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

double reductionSigma2(int size, double *inData)
{

	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(SIGMA2, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(MEAN, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result += outData_host[i];
	}
	result /= s;

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

double reductionSigma4(int size, double *inData)
{

	int cpuFinalThreshold = 256;
	int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(SIGMA4, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(MEAN, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result += outData_host[i];
	}
	result /= s;

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}
