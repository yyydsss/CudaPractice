
// command
// ncu --set full -o /home/yds/1_reduce/reduce_0_report ./1_reduce/reduce_0

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <functional> // 用于 std::function
#include "utils.hpp"

#define THREAD_PER_BLOCK 1024

// CUDA归约操作
__global__ void reduce0(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程加载一个元素到共享内存
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // 在共享内存中进行归约操作
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将线程块的归约结果写入全局内存
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }
}
//warp divergence 减少分支等待时间
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];


    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    for(unsigned int s=1; s<blockDim.x; s*=2){
        int index = 2*s*tid;
        if(index < blockDim.x){
            sdata[index]+=sdata[index+s];
        }
        __syncthreads();
    }
    
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

//bank conflict
__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

//减少idle线程
__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];


    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();


    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}


__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

//同一个warp内不需要显示同步等待
__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];


    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();


    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    

    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}




// 封装 CUDA 内核执行的函数
void run_kernel(std::function<void(float*, float*)> kernel, int N, int blockSize, int gridSize) {
    // 分配主机内存
    float *h_in = new float[N];
    float *h_out = new float[gridSize];

    // 初始化输入数组
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f; // 假设所有元素为 1.0
    }

    // 分配设备内存
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, gridSize * sizeof(float));

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // 将数据从主机复制到设备
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 执行内核 100 次，并计算平均用时
    float totalKernelTime = 0.0f;
    for (int i = 0; i < 5; i++) {
        // 记录内核执行开始时间
        cudaEventRecord(start);

        // 调用 CUDA 内核函数
        kernel(d_in, d_out);

        // 记录内核执行结束时间
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // 计算单次内核执行时间
        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, start, stop);
        totalKernelTime += kernelTime;
    }

    // 计算平均内核执行时间
    float avgKernelTime = totalKernelTime; 

    // 将结果从设备复制回主机
    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);


    // 在主机上进行最终的归约
    float final_result = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        final_result += h_out[i];
    }

    // 输出结果
    std::cout << "Final result: " << final_result << std::endl;

    // 输出计算用时
    std::cout << "Average Kernel Execution Time: " << avgKernelTime << " ms" << std::endl;


    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放设备内存
    cudaFree(d_in);
    cudaFree(d_out);

    // 释放主机内存
    delete[] h_in;
    delete[] h_out;
}

int main() {
    // 定义数组大小
    const int N = 1024*16; // 假设数组大小为 1024
    const int blockSize = THREAD_PER_BLOCK;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 定义内核函数
    auto kernel0 = [gridSize, blockSize](float *d_in, float *d_out) {
        reduce0<<<gridSize, blockSize>>>(d_in, d_out);
    };
    auto kernel1 = [gridSize, blockSize](float *d_in, float *d_out) {
        reduce1<<<gridSize, blockSize>>>(d_in, d_out);
    };
    auto kernel2 = [gridSize, blockSize](float *d_in, float *d_out) {
        reduce2<<<gridSize, blockSize>>>(d_in, d_out);
    };
    auto kernel3 = [gridSize, blockSize,N](float *d_in, float *d_out) {
        int g = (N + (2*blockSize) - 1) / (2*blockSize);
        reduce3<<<g, blockSize>>>(d_in, d_out);
    };

    auto kernel4 = [gridSize, blockSize,N](float *d_in, float *d_out) {
        int g = (N + (2*blockSize) - 1) / (2*blockSize);
        reduce4<<<g, blockSize>>>(d_in, d_out);
    };
    
    // 调用封装函数
    run_kernel(kernel0, N, blockSize, gridSize);
    run_kernel(kernel1, N, blockSize, gridSize);
    run_kernel(kernel2, N, blockSize, gridSize);
    run_kernel(kernel3, N, blockSize, gridSize);
    run_kernel(kernel4, N, blockSize, gridSize);
    return 0;
}