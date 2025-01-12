#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: Reduce Sum
__global__ void reduceSum(int *input, int *output, int n) {
    extern __shared__ int sharedData[];

    // 线程索引
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存
    sharedData[tid] = (index < n) ? input[index] : 0;
    __syncthreads();

    // 在共享内存中进行归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // 将每个块的结果写入输出数组
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    const int n = 1 << 20; // 数组大小 (1M 元素)
    const int blockSize = 256; // 线程块大小
    const int gridSize = (n + blockSize - 1) / blockSize; // 网格大小

    // 分配主机内存
    int *h_input = new int[n];
    int *h_output = new int[gridSize];

    // 初始化输入数组
    for (int i = 0; i < n; i++) {
        h_input[i] = 1; // 所有元素初始化为 1
    }

    // 分配设备内存
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, gridSize * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 CUDA 核函数
    reduceSum<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n);

    // 将结果从设备复制回主机
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 在主机上进行最终的归约求和
    int finalSum = 0;
    for (int i = 0; i < gridSize; i++) {
        finalSum += h_output[i];
    }

    // 输出结果
    std::cout << "Final Sum: " << finalSum << std::endl;

    // 释放内存
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}