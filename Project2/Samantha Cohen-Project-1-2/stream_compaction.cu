#include <cuda_runtime.h>
#include <stdio.h>
#include "CPU_stream_compaction.h"

/*__global__ void scan_workefficient(float *g_odata, float *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float temp[];

    int thid = threadIdx.x;

    int offset = 1;

    // Cache the computational window in shared memory
    temp[2*thid]   = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        temp[n - 1] = 0;
    }   

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[2*thid]   = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}

__global__ void prefix_sum_exclusive_all_lengths_kernel(float *in, float *out, int n)
{
    extern __shared__ float temp[];
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
    out[2*tx+1] = temp[2*tx+1];
}
*/

__global__ void prefix_sum_exclusive_all_lengths_kernel(float *in, float *out, int n)
{
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
	cudaFree(in);
	cudaFree(out);
}

void prefix_sum_exclusive_all_lengths(float* in1, float* out1, int width) {
	unsigned int eltNum = 512;
	int extra_space = 0;
	const unsigned int threadNum = eltNum / 2;
	const unsigned int sharedMemorySize = sizeof(float) * eltNum;
	int size = 1 * width * sizeof(float);
	float *in;
	float *out;
	int blockNum = width/eltNum;
	blockNum += 1;

	//Original Array
	cudaMalloc((void**)&in, size);
	cudaMemcpy(in,in1,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, size);

	dim3 dimBlock(threadNum*2, 1, 1);
	dim3 dimGrid(1, 1, 1);

	//1) Create one float array for every block from the input array
	vector<float*> v;
	v.resize(blockNum);
	for (int i = 0; i < blockNum; i++) {
		float t[eltNum];
		int k = 0;
		for (int j = i*eltNum; j < blockNum*eltNum; j++) {
			if (j < width) {
				t[k] = in[j];
				k++;
			}
		}
		v[i] = t;
	}

	//2) Perform inclusive prefix sum for all blocks of array values
	for (int i = 0; i < v.size(); i++) {
		prefix_sum_inclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum); //need to deal with memalloc for arrays and actual array vals
		for (int j = 0; j < width; j++) {
			v[i] = out[j];
		}
	}
	//3) Create new array of maxes
	float* maxes[blockNum];
	for (int i = 0; i < blockNum; i++) {
		maxes[i] = v[i][eltNum-1];
	}
	//4) Perform exclusive prefix sum on maxes
	for (int i = 0; i < blockNum; i++) {
		prefix_sum_exclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(maxes, out, eltNum); //need to deal with memalloc for arrays and actual array vals
	}
	//5) Add the associated value, arr[i] to each block, i, of prefix values
	for (int i = 0; i < blockNum; i++) {
		for (int j = 0; j < eltNum; j++) {
			v[i][j] += out[i]; //need to deal with memalloc etc. for arrays
		}
	}
	//6) Combine prefix blocks into 1 array
	for (int i = 0; i < blockNum; i++) {
		for (int j = 0; j < eltNum; j++) {
			out[i*eltNum+j] = v[i][j]; //need to deal with memalloc etc. for arrays
		}
	}

	prefix_sum_inclusive_one_block_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);

	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);
}
/*void prefix_sum_exclusive_all_lengths(float* in1, float* out1, int width) {
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

//	prefix_sum_exclusive_all_lengths_kernel<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(in, out, eltNum);
	scan_workefficient<<<dimGrid, dimBlock, 2 * sharedMemorySize>>>(out, in, eltNum);

	cudaMemcpy(out1, out, size, cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);
}
*/

int main () {
	//Part 1
	printf("Part 1");
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
	printf("Prefix Sum Inclusive:\n");
	printf("Input Array: \n["); printf("%f ",arr[0]); printf("%f ",arr[1]); printf("%f ",arr[2]); printf("%f ",arr[3]);
	printf("%f ",arr[4]); printf("%f ",arr[5]); printf("]\n");
	printf("Output Array: \n["); printf("%f ",ex[0]); printf("%f ",ex[1]); printf("%f ",ex[2]); printf("%f ",ex[3]);
	printf("%f ",ex[4]); printf("%f ",ex[5]); printf("]\n");

	//Part 2

	//Part 3
	//Part 3a
	printf("Part 3a\n");
	float * in1 = new float[6];
	in1[0] = 3; in1[1] = 4; in1[2] = 6; in1[3] = 7; in1[4] = 9; in1[5] = 10;
	float * out1 = new float[6];
	out1[0] = 0; out1[1] = 0; out1[2] = 0; out1[3] = 0; out1[4] = 0; out1[5] = 0;

	int size = 1*6*sizeof(float);
	int numBlocks = 1;
	dim3 threadsPerBlock(1,1);

	prefix_sum_exclusive_one_block(in1,out1,6);

	printf("GPU Prefix Sum Exclusive One Block:\n");
	printf("Input Array:\n[");
	printf("%f ",in1[0]); printf("%f ",in1[1]); printf("%f ",in1[2]); printf("%f ",in1[3]); printf("%f ",in1[4]); printf("\n");
	printf("%f ",in1[5]); printf("]\n");
	printf("Output Array:\n[");
	printf("%f ",out1[0]); printf("%f ",out1[1]); printf("%f ",out1[2]); printf("%f ",out1[3]); printf("%f ",out1[4]); printf("\n");
	printf("%f ",out1[5]); printf("]\n");
	printf("\n");

	float * in3 = new float[6];
	in3[0] = 3; in3[1] = 4; in3[2] = 6; in3[3] = 7; in3[4] = 9; in3[5] = 10;
	float * out3 = new float[6];
	out3[0] = 0; out3[1] = 0; out3[2] = 0; out3[3] = 0; out3[4] = 0; out3[5] = 0;

	prefix_sum_inclusive_one_block(in3,out3,6);

	printf("GPU Prefix Sum Inclusive One Block:\n");
	printf("Input Array:\n[");
	printf("%f ",in3[0]); printf("%f ",in3[1]); printf("%f ",in3[2]); printf("%f ",in3[3]); printf("%f ",in3[4]); printf("\n");
	printf("%f ",in3[5]); printf("]\n");
	printf("Output Array:\n[");
	printf("%f ",out3[0]); printf("%f ",out3[1]); printf("%f ",out3[2]); printf("%f ",out3[3]); printf("%f ",out3[4]); printf("\n");
	printf("%f ",out3[5]); printf("]\n");
	printf("\n");

	//Part 3b
	printf("Part 3b\n"); //ONLY GOES TO 512 SO FAR
	int size2 = 2000;
	float * in2 = new float[size2];
	float * out2 = new float[size2];
	for (int i = 0; i < size2; i++) {
		in2[i] = 1;
		out2[i] = 0;
	}

	printf("%i ", 4/3);
	printf("%f ", 4.0f/3.0f);

	//prefix_sum_exclusive_all_lengths(in2,out2,size2);
	//prefix_sum_exclusive_one_block(in2,out2,size2);

	/*printf("GPU Prefix Sum Exclusive All Lengths:\n");
	printf("Input Array:\n[");
	for (int i = 0; i < size2; i++) {
		printf("%f ",in2[i]);
	}
	printf("]\n");
	printf("Output Array:\n[");
	for (int i = 0; i < size2; i++) {
		printf("%f ",out2[i]);
	}
	printf("]\n");*/



	getchar();
	return 0;
}