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
__global__ void ray_shoot(int maxX, int maxY, float rsrc, float ldc, float xsrc, float ysrc, float lens_scale, float *xlens, float *ylens, float*eps, int num_lenses, float *dev_arr)
{
	int threadBlockPos = (blockIdx.x * blockDim.x) + threadIdx.x;

	int y = threadBlockPos / maxY;
	int x = threadBlockPos - (maxX * y);

	const float rsrc2 = rsrc * rsrc; 

	float xl = XL1 + x * lens_scale;
	float yl = YL1 + y * lens_scale; 
	float xs = XL2 + x * lens_scale; 
	float ys = YL2 + y * lens_scale;

	float dx, dy, dr;

	xs = xl; 
	ys = yl;

	for(int p = 0; p < num_lenses; ++p){
		dx = xl - xlens[p];
	    dy = yl - ylens[p];
	    dr = dx * dx + dy * dy;
	    xs -= eps[p] * dx / dr;
	    ys -= eps[p] * dy / dr;
	}

	float xd = xs - xsrc; 
	float yd = ys - ysrc; 
	float sep2 = xd * xd + yd * yd; 
	if(sep2 < rsrc2){
		float mu = sqrt(1-sep2/rsrc2); 
		//somehow need to copy the info into an array which is then sent back to the host memeory
		dev_arr[(y*maxX) + x] = 1 - ldc * (1-mu);
		//std::cout << dev_arr[(y*maxX) + x] << std::endl;
		//dev_lensim(y, x) = 1.0 - ldc * (1-mu);
	}

	// Source star parameters. You can adjust these if you like - it is
	// interesting to look at the different lens images that result
	//const float rsrc = 0.1;      // radius
	//const float ldc  = 0.5;      // limb darkening coefficient
	//const float xsrc = 0.0;      // x and y centre on the map
	//const float ysrc = 0.0;

	//const float rsrc2 = rsrc * rsrc;

	//int x = blockDim.x * blockIdx.x + threadIdx.x;
	//int y = blockDim.y * blockIdx.y + threadIdx.y;


	//XL1, YL1, XL2, YL2
	//ix, iy (based on x and y)
	//xlens, ylens, eps, nlenses
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
	const int nlenses = set_example_1(&xlens, &ylens, &eps);
	std::cout << "# Simulating " << nlenses << " lens system" << std::endl;

	// Source star parameters. You can adjust these if you like - it is
	// interesting to look at the different lens images that result
	const float rsrc = 0.1;      // radius
	const float ldc  = 0.5;      // limb darkening coefficient
	const float xsrc = 0.0;      // x and y centre on the map
	const float ysrc = 0.0;

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

	int total_pixels = npixx + npixy;
		
	int threadsPerBlock = 1024;
	int numBlocks = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

	//setup the array that will be sent to device for all of the pixels and will eventually be retrieved
	float *arr_lensim = (float*)malloc(sizeof(float)*npixx*npixy);

	for(int y = 0; y < npixy; y++){
		for(int x = 0; x < npixx; x++){
			arr_lensim[(y * npixx) + x] = 0;
		}
	}

	//--------------------------
	//cuda part
	//==========================

	float *dev_xlens;
	float *dev_ylens; 
	float *dev_eps;
	float *dev_arr_lensim;
	int size = sizeof(float) * nlenses;

	

	cudaError_t err = cudaMalloc((void**)&dev_xlens, size);
	err_check(err);
	err = cudaMalloc((void**)&dev_ylens, size);
	err_check(err);
	err = cudaMalloc((void**)&dev_eps, size);
	err_check(err);
	err = cudaMalloc((void**)&dev_arr_lensim, sizeof(float)*npixx*npixy);
	err_check(err);

	err = cudaMemcpy(dev_xlens, xlens, size, cudaMemcpyHostToDevice);
	err_check(err);
	err = cudaMemcpy(dev_ylens, ylens, size, cudaMemcpyHostToDevice);
	err_check(err);
	err = cudaMemcpy(dev_eps, eps, size, cudaMemcpyHostToDevice);
	err_check(err);
	err = cudaMemcpy(dev_arr_lensim, arr_lensim, sizeof(float)*npixx*npixy, cudaMemcpyHostToDevice);
	err_check(err);

	//=====================================
	//=====================================
	//Need to create a new array on both host and device memory and then copy from device the new values and then convert it back to the Array object type so that we can use dump_array();
	
	//mem alloc maxX and maxY and then add to the function below
	//int maxX, int maxY, float rsrc, float ldc, float xsrc, float ysrc, float lens_scale, float *xlens, float *ylens, float*eps, int num_lenses
	ray_shoot<<<numBlocks, threadsPerBlock>>>(npixx, npixy, rsrc, ldc, xsrc, ysrc, lens_scale, dev_xlens, dev_ylens, dev_eps, nlenses, dev_arr_lensim);


	err = cudaMemcpy(arr_lensim, dev_arr_lensim, sizeof(float)*npixx*npixy, cudaMemcpyDeviceToHost);
	err_check(err);


	// for(int y = 0; y < npixy; y++){
	// 	for(int x = 0; x < npixx; x++){
	// 		//if(arr_lensim[(y*npixy)+x] != 0.0){
	// 			std::cout << arr_lensim[(y*npixx) + x] << " ";
	// 		//}
	// 	}
	// 	//std::cout << std::endl;
	// }

	cudaFree(dev_xlens); 
	cudaFree(dev_ylens); 
	cudaFree(dev_eps);
	cudaFree(dev_arr_lensim);

	clock_t tend = clock();
	double tms = diffclock(tend, tstart);
	std::cout << "# Time elapsed: " << tms << " ms " << std::endl;

	// Write the lens image to a FITS formatted file. You can view this
	// image file using ds9
	dump_array<float, 2>(lensim, "lens.fit");

	delete[] xlens;
	delete[] ylens;
	delete[] eps;
}
