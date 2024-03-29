#include "stdafx.h"

void interface(double &precision, double &tau, double &g, size_t &N, double &lambda, std::string &filename, bool &isLoad);
void save(std::clock_t &timer, const double timerLimit, systemEquCuda_3D &S);

int main()
{	
	bool isDebug = true;

	std::clock_t start, startSave;
	double duration;
	start = std::clock();
	startSave = std::clock();
	
	std::ofstream out_maxVal("out_maxVal.txt");
	out_maxVal.precision(14);

	double timeLimitSave;
	double precision, tau, lambda, g;
	size_t N;
	std::string filename;
	bool isLoad;

	if (isDebug)
	{
		precision = 0.05;
		tau = 0.5;
		N = 100;
		lambda = -0.2;
		g = 0.02;
		timeLimitSave = 30000;
		filename = "in.txt";
		isLoad = false;
	}
	else
	{
		interface(precision, tau, g, N, lambda, filename, isLoad);
		timeLimitSave = 1800;
	}


	systemEquCuda_3D S(filename, precision, tau, lambda, g, isLoad);
	std::cout << "--- Program GPU started --- \n";

	S.printingVTK();
	S.printingVTKrho();
	S.printingMaxVal(out_maxVal);

	for (size_t i = 0; i < N; i++)
	{
		std::cout << "Step started...\t";
		S.evaluate();
		std::cout << "Step completed\nPrinting started...\t";

		S.printingVTK();
		S.printingVTKrho();
		S.printingMaxVal(out_maxVal);
		save(startSave, timeLimitSave, S);

		std::cout << "Printing done\n";
		printf("Energy = %.4e\tDeltaE = %.4e\n", S.get_energy(), S.get_delta());

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "step: " << i << "\t\tcalculated time: " << S.get_time()  << "\t\texecution time: " << duration << std::endl << std::endl;
		std::cout << "------------------------------------------------------------------\n\n";
	}
	
	out_maxVal.close();
	std::cin >> tau;
	return 0;
}

void interface(double &precision, double &tau, double &g, size_t &N, double &lambda, std::string &filename, bool &isLoad)
{
	int a;
	std::cout << "1. New task\n";
	std::cout << "2. Load task\n";

	bool isDone = false;
	do
	{
		std::cin >> a;
		switch (a)
		{
		case 1:
			isLoad = false;
			filename = "in.txt";
			std::cout << "lambda = ";
			std::cin >> lambda;
			std::cout << "g = ";
			std::cin >> g;
			isDone = true;
			break;
		case 2:
			isLoad = true;
			filename = "saveGrid.asv";
			isDone = true;
			break;
		default:

			break;
		}
	} while (!isDone);
	
	std::cout << "precision = ";
	std::cin >> precision;
	std::cout << "tau = ";
	std::cin >> tau;
	std::cout << "N = ";
	std::cin >> N;

}

void save(std::clock_t &timer, const double timerLimit, systemEquCuda_3D &S)
{
	double duration = (std::clock() - timer) / (double)CLOCKS_PER_SEC;
	if (duration >= timerLimit)
	{
		std::cout << "\n\nSAVING...\t";
		S.save();
		timer = std::clock();
		std::cout << "SAVED\n\n";
	}
}

//cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//cudaEventRecord(stop, 0); cudaEventSynchronize(stop);	float elapsedTime;	cudaEventElapsedTime(&elapsedTime, start, stop);
//std::cout << "cuFFT time: " << elapsedTime << "ms \n";	cudaEventDestroy(start); cudaEventDestroy(stop);

//std::clock_t startCPU;	double duration;	startCPU = std::clock();
//duration = (std::clock() - startCPU) / (double)CLOCKS_PER_SEC;	std::cout << "FFTW cpu time: " << duration*1000 << "ms \n";

