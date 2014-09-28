#include <cuda_runtime.h>
#include <stdio.h>
#include "CPU_stream_compaction.h"
#include <thrust/device_vector.h>
#define NUM_BANKS 16; 
#define LOG_NUM_BANKS 4; 
#define CONFLICT_FREE_OFFSET(n) / ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS));  

__global__ void prefix_sum_exclusive_work_efficient_kernel(float *in, float *out, int n)
{
   /* extern __shared__ float temp[];
    int tx = threadIdx.x;
    int offset = 1;
	int a = tx;  
	int b = tx + (n/2);  
	int bankOffsetA = CONFLICT_FREE_OFFSET(a); 
	int bankOffsetB = CONFLICT_FREE_OFFSET(b); 
	temp[a + bankOffsetA] = in[a];  
	temp[b + bankOffsetB] = in[b];  
    for (int i = n>>1; i > 0; i >>= 1) {
        __syncthreads();
        if (tx < i) {
            a = offset*(2*tx+1)-1;
            b = offset*(2*tx+2)-1;
			a += CONFLICT_FREE_OFFSET(a);  
			b += CONFLICT_FREE_OFFSET(b); 
            temp[b] += temp[a];
        }
        offset *= 2;
    }
	if (tx == 0) { 
		temp[n-1+CONFLICT_FREE_OFFSET(n-1)] = 0;
	} 
    for (int i = 1; i < n; i *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tx < i) {
            a = offset*(2*tx+1)-1;
            b = offset*(2*tx+2)-1;
			a += CONFLICT_FREE_OFFSET(a);  
			b += CONFLICT_FREE_OFFSET(b); 
            float t   = temp[a];
            temp[a]  = temp[b];
            temp[b] += t;
        }
    }
    __syncthreads();
	out[a] = temp[a+bankOffsetA];  
	out[b] = temp[b+bankOffsetB];*/
}

__global__ void prefix_sum_exclusive_all_lengths_kernel(float *in, float *out, int n)
{
	int tx = threadIdx.x;
	out[tx] = in[tx];
    /*extern __shared__ float temp[];
    int tx = threadIdx.x;
    int offset = 1;
    temp[2*tx]   = in[2*tx];
    temp[2*tx+1] = in[2*tx+1];
    for (int i = n>>1; i > 0; i >>= 1) {
        __syncthreads();
        if (tx < i) {
            int ai = offset*(2*tx+1)-1;
            int bi = offset*(2*tx+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (tx == 0) {
		temp[n-1] = 0;
    }   
    for (int i = 1; i < n; i *= 2) {
        offset >>= 1;
        __syncthreads();

        if (tx < i) {
            int ai = i*(2*tx+1)-1;
            int bi = i*(2*tx+2)-1;

            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    out[2*tx]   = temp[2*tx];
    out[2*tx+1] = temp[2*tx+1];*/
}

__global__ void prefix_sum_exclusive_one_block_kernel(float* in, float* out, int n) {
	extern __shared__ float temp[]; 
	int tx = threadIdx.x;  
	int pout = 0;
	int pin = 1;
	if (tx > 0) {
		temp[pout*n+tx] = in[tx-1];
	} else {
		temp[pout*n+tx] = 0; 
	}
	for (int i = 1; i < n; i *= 2)  {  
	  pout = 1 - pout;
	  pin = 1 - pout;  
	__syncthreads();  
		temp[pout*n+tx] = temp[pin*n+tx];
	  if (tx >= i) { 
		temp[pout*n+tx] += temp[pin*n+tx-i];  
	  }
	}  
	__syncthreads();  
	out[tx] = temp[pout*n+tx];
}

__global__ void prefix_sum_inclusive_one_block_kernel(float* in, float* out, int n) {
	extern __shared__ float temp[]; 
	int tx = threadIdx.x;  
	int pout = 0;
	int pin = 1;
	temp[pout*n+tx] = in[tx];
	for (int i = 1; i < n; i *= 2)  {  
	  pout = 1 - pout;
	  pin = 1 - pout;  
	  __syncthreads();  
		temp[pout*n+tx] = temp[pin*n+tx];
	  if (tx >= i) { 
		temp[pout*n+tx] += temp[pin*n+tx-i];  
	  }
	}  
	__syncthreads();  
	out[tx] = temp[pout*n+tx];
} 

__global__ void prefix_sum_exclusive_naive_kernel(float* in, float* out, int n, int k, int k2) {
	int tx = threadIdx.x;  
	int pout = 0;
	int pin = 1;
	out[pout*n+tx] = in[tx];
	for (int i = 1; i < n; i *= 2)  {  
	  pout = 1 - pout;
	  pin = 1 - pout;  
	  __syncthreads();  
		out[pout*n+tx] = out[pin*n+tx];
	  if (tx >= i) { 
		out[pout*n+tx] += out[pin*n+tx-i];  
	  }
	}  
	__syncthreads(); 
	out[tx+1] = out[pout*n+tx];
	out[0] = 0;
} 

void prefix_sum_exclusive_naive(float* in1, float* out1, int width) {
	unsigned int eltNum = 512;
	int extra_space = 0;
	const unsigned int threadNum = eltNum / 2;
	const unsigned int sharedMemorySize = sizeof(float) * eltNum;
	int size = 1 * width * sizeof(float);
	float *in;
	float *out;
	//Original Array
	cudaMalloc((void**)&in, size);
	cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, size);

	dim3 dimBlock(threadNum*2, 1, 1);
	dim3 dimGrid(1, 1, 1);

	for (int d = 1; d < log(width*1.0f); d++) {
		for (int k = 0; k < width; k++) {
			if (k >= powf(2.0f, d-1.0f)) {
				int k2 = k-powf(2.0f, d-1.0f);
				prefix_sum_exclusive_naive_kernel<<<dimGrid, dimBlock>>>(in, out, eltNum, k, k2);
			}
		}
	}
	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);
}

void prefix_sum_exclusive_one_block(float* in1, float* out1, int width) {
	unsigned int eltNum = 512;
	int extra_space = 0;
	const unsigned int threadNum = eltNum / 2;
	const unsigned int sharedMemorySize = sizeof(float) * eltNum;
	int size = 1 * width * sizeof(float);
	float *in;
	float *out;
	//Original Array
	cudaMalloc((void**)&in, size);
	cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, size);

	dim3 dimBlock(threadNum*2, 1, 1);
	dim3 dimGrid(1, 1, 1);

	prefix_sum_exclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);

	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);
}

void prefix_sum_inclusive_one_block(float* in1, float* out1, int width) {
	unsigned int eltNum = 512;
	int extra_space = 0;
	const unsigned int threadNum = eltNum / 2;
	const unsigned int sharedMemorySize = sizeof(float) * eltNum;
	int size = 1 * width * sizeof(float);
	float *in;
	float *out;
	//Original Array
	cudaMalloc((void**)&in, size);
	cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, size);

	dim3 dimBlock(threadNum*2, 1, 1);
	dim3 dimGrid(1, 1, 1);

	prefix_sum_inclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);

	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	/*printf("%f\n",out1[1]);
	printf("%f\n",out[1]);*/
	cudaFree(in);
	cudaFree(out);
}

void prefix_sum_exclusive_all_lengths(float* in1, float* out1, int width) {
	const unsigned int eltNum = 512;
	if (width > eltNum) {
		int extra_space = 0;
		const unsigned int threadNum = eltNum / 2;
		const unsigned int sharedMemorySize = sizeof(float) * eltNum;
		int size = 1 * width * sizeof(float);
		float *in;
		float* out;
		const unsigned int blockNum = (width/eltNum) + 1;

		//Original Array
		cudaMalloc((void**)&in, size);
		cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&out, size);

		dim3 dimBlock(threadNum*2, 1, 1);
		dim3 dimGrid(1, 1, 1);

		//1) Create one float array for every block from the input array
		thrust::device_vector<float> v;
		v.resize(width);
		for (int i = 0; i < blockNum; i++) {
			//float t[eltNum];
			//int k = 0;
			for (int j = i*eltNum; j < (i+1)*eltNum; j++) {
				if (j < width) {
					//t[k] = in1[j];
					//k++;
					v[j] = in1[j];
				} 
			}
			//v[i] = t;
		}

		//2) Perform inclusive prefix sum for all blocks of array values
		for (int i = 0; i < blockNum; i++) {
			//float* t1 = v[i];
			float* t1 = new float[eltNum];
			int k = 0;
			for (int j = i*eltNum; j < (1+i)*eltNum; j++) {
				if (j < width && k < eltNum) {
					t1[k] = v[j];
				} else {
					t1[k] = 0;
				}
				k++;
			}
			float* t;
			int size1 = 1 * eltNum * sizeof(float);
			cudaMalloc((void**)&t, size1);
			cudaMemcpy(t,t1,size,cudaMemcpyHostToDevice);
			float to1[eltNum];
			float* to;
			cudaMalloc((void**)&to, size1);
			prefix_sum_inclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(t, to, eltNum); //need to deal with memalloc for arrays and actual array vals
			cudaMemcpy(to1, to, size, cudaMemcpyDeviceToHost);
			int l = 0;
			for (int j = i*eltNum; j < (1+i)*eltNum; j++) {
				if (j < width && l < eltNum) {
					v[j] = to1[l];
				}
				l++;
			}
			//v[i] = to;
			cudaFree(t);
			cudaFree(to);
		}

		//3) Create new array of maxes
		float* maxes1 = new float[blockNum];
		//float* tmp = new float[eltNum];
		for (int i = 0; i < blockNum; i++) {
			//tmp = v[i];
			//maxes1[i] = tmp[eltNum-1]; //You have an issue here
			if ((i+1)*eltNum-1 < width) {
				maxes1[i] = v[(i+1)*eltNum-1];
			} else {
				maxes1[i] = v[width-1];
			}
		}

		//4) Perform exclusive prefix sum on maxes
		float* maxes;
		int size2 = 1 * blockNum * sizeof(float);
		cudaMalloc((void**)&maxes, size2);
		cudaMemcpy(maxes,maxes1,size,cudaMemcpyHostToDevice);
		float* maxeso1 = new float[eltNum];
		float* maxeso;
		cudaMalloc((void**)&maxeso, size);
		prefix_sum_exclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(maxes, maxeso, eltNum); //need to deal with memalloc for arrays and actual array vals
		cudaMemcpy(maxeso1, maxeso, size2, cudaMemcpyDeviceToHost);
		
		//5) Add the associated value, arr[i] to each block, i, of prefix values
		for (int i = 0; i < blockNum; i++) {
			for (int j = i; j < (i+1)*eltNum; j++) {
				if (j < width) {
					v[j] = v[j] + maxeso1[i]; //need to deal with memalloc etc. for arrays
				}
			}
		}
		cudaFree(maxes);
		cudaFree(maxeso);

		//6) Combine prefix blocks into 1 array
		float* tempy = new float[width];
		for (int i = 0; i < blockNum; i++) {
			for (int j = i; j < (i+1)*eltNum; j++) {
				if (j < width) {
					tempy[j] = v[j]; //need to deal with memalloc etc. for arrays
				}
			}
		}
		
		cudaMalloc((void**)&tempy, size);
		//float to1[eltNum];
		//float* to;
		//cudaMalloc((void**)&to, size1);
		prefix_sum_exclusive_all_lengths_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(tempy/*t*/, out/*to*/, eltNum); //need to deal with memalloc for arrays and actual array vals
		//cudaMemcpy(to1, to, size, cudaMemcpyDeviceToHost);

		cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
		cudaFree(in);
		cudaFree(out);
		std::vector<int> stl_vector(v.size());
		thrust::copy(v.begin(), v.end(), stl_vector.begin());
		for (int i = 0; i < stl_vector.size(); i++) {
			printf("%f ",stl_vector[i]);
		}
	} else {
			prefix_sum_inclusive_one_block(in1, out1, width);
	}
}

void prefix_sum_exclusive_work_efficient(float* in1, float* out1, int width) {
	unsigned int eltNum = 512;
	int extra_space = 0;
	const unsigned int threadNum = eltNum / 4;
	const unsigned int sharedMemorySize = sizeof(float) * eltNum;
	int size = 1 * width * sizeof(float);
	float *in;
	float *out;
	//Original Array
	cudaMalloc((void**)&in, size);
	cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, size);

	dim3 dimBlock(threadNum*2, 1, 1);
	dim3 dimGrid(2, 2, 1);

	prefix_sum_exclusive_work_efficient_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);

	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);
}

__global__ void compact_stream_bool_kernel(float* in, float* out, int n) {
	int tx = threadIdx.x;
	if (in[tx] > 0) {
		out[tx] = 1;
	} else {
		out[tx] = 0;
	}
} 

void stream_compact(float* in1, float* out1, int width) {
	unsigned int eltNum = 512;
	int extra_space = 0;
	const unsigned int threadNum = eltNum / 4;
	const unsigned int sharedMemorySize = sizeof(float) * eltNum;
	int size = 1 * width * sizeof(float);
	float *in;
	float *out;
	//Original Array
	cudaMalloc((void**)&in, size);
	cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, size);

	dim3 dimBlock(threadNum*2, 1, 1);
	dim3 dimGrid(2, 2, 1);

	compact_stream_bool_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);
	in = out;
	prefix_sum_exclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);
	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);
}

int main () {
	//Part 1
	printf("Part 1\n");
	CPU_stream_compaction sc;
	float arr[6];
	arr[0] = 3.0f;
	arr[1] = 4.0f;
	arr[2] = 6.0f;
	arr[3] = 7.0f;
	arr[4] = 9.0f;
	arr[5] = 10.0f;
	float* in = sc.CPU_prefix_sum_inclusive(arr, 6);
	printf("Prefix Sum Inclusive:\n");
	printf("Input Array: \n["); printf("%f ",arr[0]); printf("%f ",arr[1]); printf("%f ",arr[2]); printf("%f ",arr[3]);
	printf("%f ",arr[4]); printf("%f ",arr[5]); printf("]\n");
	printf("Output Array: \n["); printf("%f ",in[0]); printf("%f ",in[1]); printf("%f ",in[2]); printf("%f ",in[3]);
	printf("%f ",in[4]); printf("%f ",in[5]); printf("]\n");

	float* ex = sc.CPU_prefix_sum_exclusive(arr, 6);
	printf("Prefix Sum Exclusive:\n");
	printf("Input Array: \n["); printf("%f ",arr[0]); printf("%f ",arr[1]); printf("%f ",arr[2]); printf("%f ",arr[3]);
	printf("%f ",arr[4]); printf("%f ",arr[5]); printf("]\n");
	printf("Output Array: \n["); printf("%f ",ex[0]); printf("%f ",ex[1]); printf("%f ",ex[2]); printf("%f ",ex[3]);
	printf("%f ",ex[4]); printf("%f ",ex[5]); printf("]\n");

	//Part 2 - NEED HELP CANNOT IMPLEMENT THIS ALGORITHM
	printf("\nPart 2\n");
	
	float * in1 = new float[6];
	in1[0] = 3; in1[1] = 4; in1[2] = 6; in1[3] = 7; in1[4] = 9; in1[5] = 10;
	float * out1 = new float[6];
	out1[0] = 0; out1[1] = 0; out1[2] = 0; out1[3] = 0; out1[4] = 0; out1[5] = 0;

	int size = 1*6*sizeof(float);
	int numBlocks = 1;
	dim3 threadsPerBlock(1,1);

	prefix_sum_exclusive_naive(in1,out1,6);

	printf("GPU Prefix Sum Exclusive Naive:\n");
	printf("Input Array:\n[");
	printf("%f ",in1[0]); printf("%f ",in1[1]); printf("%f ",in1[2]); printf("%f ",in1[3]); printf("%f ",in1[4]); printf("\n");
	printf("%f ",in1[5]); printf("]\n");
	printf("Output Array:\n[");
	printf("%f ",out1[0]); printf("%f ",out1[1]); printf("%f ",out1[2]); printf("%f ",out1[3]); printf("%f ",out1[4]); printf("\n");
	printf("%f ",out1[5]); printf("]\n");
	printf("\n");

	//Part 3
	//Part 3a
	printf("Part 3a\n");

	prefix_sum_exclusive_one_block(in1,out1,6);

	printf("GPU Prefix Sum Exclusive One Block:\n");
	printf("Input Array:\n[");
	printf("%f ",in1[0]); printf("%f ",in1[1]); printf("%f ",in1[2]); printf("%f ",in1[3]); printf("%f ",in1[4]); printf("\n");
	printf("%f ",in1[5]); printf("]\n");
	printf("Output Array:\n[");
	printf("%f ",out1[0]); printf("%f ",out1[1]); printf("%f ",out1[2]); printf("%f ",out1[3]); printf("%f ",out1[4]); printf("\n");
	printf("%f ",out1[5]); printf("]\n");
	printf("\n");

	prefix_sum_inclusive_one_block(in1,out1,6);

	printf("GPU Prefix Sum Inclusive One Block:\n");
	printf("Input Array:\n[");
	printf("%f ",in1[0]); printf("%f ",in1[1]); printf("%f ",in1[2]); printf("%f ",in1[3]); printf("%f ",in1[4]); printf("\n");
	printf("%f ",in1[5]); printf("]\n");
	printf("Output Array:\n[");
	printf("%f ",out1[0]); printf("%f ",out1[1]); printf("%f ",out1[2]); printf("%f ",out1[3]); printf("%f ",out1[4]); printf("\n");
	printf("%f ",out1[5]); printf("]\n");
	printf("\n");

	//Part 3b - NEED HELP CANNOT GET VECTORS WITH FLOAT* TO WORK
	printf("Part 3b\n");
	int size2 = 550;
	float * in2 = new float[size2];
	float * out2 = new float[size2];//CUDAMALLOC YOUR ARRAYS
	//in2 = (float*) malloc(size);
	//out2 = (float*) malloc(size);
	for (int i = 0; i < size2; i++) {
		in2[i] = 1;
		out2[i] = 0;
	}

	//prefix_sum_exclusive_all_lengths(in2,out2,size2);
	//free(in2);
	//free(out2);
	printf("GPU Prefix Sum Exclusive All Lengths:\n");
	/*printf("Input Array:\n[");
	for (int i = 0; i < size2; i++) {
		printf("%f ",in2[i]);
	}
	printf("]\n");
	printf("Output Array:\n[");
	for (int i = 0; i < size2; i++) {
		printf("%f ",out2[i]);
	}
	printf("]\n");*/

	//Part 4
	//CPU
	printf("\nPart 4\n");
	printf("CPU Scatter:");
	float arr2[9];
	arr2[0] = 3.0f;
	arr2[1] = 0.0f;
	arr2[2] = 4.0f;
	arr2[3] = 0.0f;
	arr2[4] = 0.0f;
	arr2[5] = 6.0f;
	arr2[6] = 7.0f;
	arr2[7] = 9.0f;
	arr2[8] = 10.0f;
	float* exx = sc.CPU_prefix_sum_exclusive(arr2, 9);
	printf("Input Array:\n[");
	for (int i = 0; i < 9; i++) {
		printf("%f ",arr2[i]);
	}
	printf("]\n");
	printf("Output Array:\n[");
	for (int i = 0; i < 6; i++) {
		printf("%f ",exx[i]);
	}
	printf("]\n");

	//GPU Scatter No Thrust
	printf("GPU Scatter:\n");
	exx = new float[9];
	stream_compact(arr2,exx,9);

	printf("Input Array:\n[");
	for (int i = 0; i < 9; i++) {
		printf("%f ",arr2[i]);
	}
	printf("]\n");
	printf("Output Array:\n[");
	for (int i = 0; i < 9; i++) {
		printf("%f ",exx[i]);
	}
	printf("]\n");
	
	
	//Extra Credit
	printf("\nExtra Credit\n");

	prefix_sum_exclusive_work_efficient(in1,out1,6);

	printf("\nGPU Prefix Sum Exclusive Work Efficient:\n");
	printf("Input Array:\n[");
	printf("%f ",in1[0]); printf("%f ",in1[1]); printf("%f ",in1[2]); printf("%f ",in1[3]); printf("%f ",in1[4]); printf("\n");
	printf("%f ",in1[5]); printf("]\n");
	printf("Output Array:\n[");
	printf("%f ",out1[0]); printf("%f ",out1[1]); printf("%f ",out1[2]); printf("%f ",out1[3]); printf("%f ",out1[4]); printf("\n");
	printf("%f ",out1[5]); printf("]\n");
	printf("\n");

	//free(in1);
	//free(out1);
	getchar();
	return 0;
}