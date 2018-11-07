/* Vector addition deom on GPU

   To compile: nvcc -o testprog1 testprog1.cu

 */
#include <ctime>
#include <iostream>
#include <string>
#include <cmath>

#include <cuda.h>
#include "lenses.h"
#include "arrayff.hxx"


// Boundaries in physical units on the lens plane
const float WL  = 2.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

// Source star parameters. You can adjust these if you like - it is
// interesting to look at the different lens images that result
const float rsrc = 0.1;      // radius
const float ldc  = 0.5;      // limb darkening coefficient
const float xsrc = 0.0;      // x and y centre on the map
const float ysrc = 0.0;

// Used to time code. OK for single threaded programs but not for
// multithreaded programs. See other demos for hints at timing CUDA
// code.
double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks = clock1 - clock2;
  double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
  return diffms; // Time difference in milliseconds
}

void err_check(cudaError_t err){
	if(err != cudaSuccess){
		std::cout << "Cuda Error: " << cudaGetErrorString(cudaGetLastError())<< std::endl;
	}
}

// Kernel that executes on the CUDA device. This is executed by ONE
// stream processor
__global__ void ray_shoot(int *maxX, int *maxY, float *lens_scale, float *xlens, float *ylens, float*eps, int *num_lenses, float *dev_arr)
{
	int threadBlockPos = (blockIdx.x * blockDim.x) + threadIdx.x;

	int y = threadBlockPos / (*maxY);
	int x = threadBlockPos - ((*maxX) * y);

	const float rsrc2 = rsrc * rsrc; 

	float xl = XL1 + x * (*lens_scale);
	float yl = YL1 + y * (*lens_scale); 
	float xs = XL2 + x * (*lens_scale); 
	float ys = YL2 + y * (*lens_scale);

	float dx, dy, dr;
	xs = xl;
	ys = yl;
	for(int p = 0; p < (*num_lenses); ++p){
		dx = xl - xlens[p];
	    dy = yl - ylens[p];
	    dr = dx * dx + dy * dy;
	    xs -= eps[p] * dx / dr;
	    ys -= eps[p] * dy / dr;
	}

	float xd = xs - xsrc; 
	float yd = ys - ysrc; 
	float sep2 = (xd * xd) + (yd * yd); 
	
	if(sep2 < rsrc2){
		float mu = sqrtf(1.0f-sep2/rsrc2); 
		dev_arr[threadBlockPos] = 1.0 - ldc * (1-mu);
	}
}

// main routine that executes on the host
int main(void)
{

	// Set up lensing system configuration - call example_1, _2, _3 or
	// _n as you wish. The positions and mass fractions of the lenses
	// are stored in these arrays
	float* xlens;
	float* ylens;
	float* eps;
	const int nlenses = set_example_2(&xlens, &ylens, &eps);
	std::cout << "# Simulating " << nlenses << " lens system" << std::endl;


	// Pixel size in physical units of the lens image. You can try finer
	// lens scale which will result in larger images (and take more
	// time).
	const float lens_scale = 0.005;

	// Size of the lens image
	const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
	const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
	std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;

	// Put the lens image in this array
  	Array<float, 2> lensim(npixy, npixx);

	clock_t tstart = clock();

	int total_pixels = npixx * npixy;
	std::cout << "total pixels: " << total_pixels << std::endl;
	int threadsPerBlock = 1024;
	std::cout << "total Threads per block: " << threadsPerBlock << std::endl;
	int numBlocks = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
	std::cout << "total blocks per grid: " << numBlocks << std::endl;

	//setup the array that will be sent to device for all of the pixels and will eventually be retrieved
	float *arr_lensim = (float*)malloc(sizeof(float)*npixx*npixy);

	for(int y = 0; y < npixy; y++){
		for(int x = 0; x < npixx; x++){
			arr_lensim[(y * npixx) + x] = 0;
		}
	}

	std::cout << "host----: " << eps[0] << std::endl;

	//--------------------------
	//cuda part
	//==========================

	int *dev_npixx;
	int *dev_npixy;
	float *dev_lens_scale;
	float *dev_xlens;
	float *dev_ylens; 
	float *dev_eps;
	int *dev_nlenses;
	float *dev_arr_lensim;

	int size = sizeof(float) * nlenses;

	cudaMalloc((void**)&dev_npixx, sizeof(int));
	cudaMalloc((void**)&dev_npixy, sizeof(int));
	cudaMalloc((void**)&dev_lens_scale, sizeof(float));
	cudaMalloc((void**)&dev_xlens, size);
	cudaMalloc((void**)&dev_ylens, size);
	cudaMalloc((void**)&dev_eps, size);
	cudaMalloc((void**)&dev_nlenses, sizeof(int));
	cudaMalloc((void**)&dev_arr_lensim, sizeof(float)*npixx*npixy);

	cudaMemcpy(dev_npixx, &npixx, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_npixy, &npixy, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lens_scale, &lens_scale, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_xlens, xlens, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ylens, ylens, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_eps, eps, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nlenses, &nlenses, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_arr_lensim, arr_lensim, sizeof(float)*npixx*npixy, cudaMemcpyHostToDevice);

	//=====================================
	//=====================================
	//Need to create a new array on both host and device memory and then copy from device the new values and then convert it back to the Array object type so that we can use dump_array();
	
	ray_shoot<<<numBlocks, threadsPerBlock>>>(dev_npixx, dev_npixy, dev_lens_scale, dev_xlens, dev_ylens, dev_eps, dev_nlenses, dev_arr_lensim);

	cudaMemcpy(arr_lensim, dev_arr_lensim, sizeof(float)*npixx*npixy, cudaMemcpyDeviceToHost);


	for(int y = 0; y < npixy; y++){
		for(int x = 0; x < npixx; x++){
			if(arr_lensim[(y*npixy)+x] != 0.0){
				lensim(y, x) = arr_lensim[(y*npixx) + x];
			}
		}
	}

	cudaFree(dev_xlens); 
	cudaFree(dev_ylens); 
	cudaFree(dev_eps);
	cudaFree(dev_arr_lensim);

	clock_t tend = clock();
	double tms = diffclock(tend, tstart);
	std::cout << "# Time elapsed: " << tms << " ms " << std::endl;

	// Write the lens image to a FITS formatted file. You can view this
	// image file using ds9
	dump_array<float, 2>(lensim, "lens2.fit");

	delete[] xlens;
	delete[] ylens;
	delete[] eps;
}
