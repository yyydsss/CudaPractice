#include <iostream>
#include <cuda_runtime.h>

// 定义 FETCH_FLOAT2 和 FETCH_FLOAT4 宏
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>((pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>((pointer))[0])

// 普通的加法核函数
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 向量化的加法核函数（float2）
__global__ void vec2_add(float* a, float* b, float* c, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    if (idx + 1 < n) {  // 确保不越界
        float2 reg_a = FETCH_FLOAT2(&a[idx]);
        float2 reg_b = FETCH_FLOAT2(&b[idx]);
        float2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        FETCH_FLOAT2(&c[idx]) = reg_c;
    }
}

// 向量化的加法核函数（float4）
__global__ void vec4_add(float* a, float* b, float* c, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (idx + 3 < n) {  // 确保不越界
        float4 reg_a = FETCH_FLOAT4(&a[idx]);
        float4 reg_b = FETCH_FLOAT4(&b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FETCH_FLOAT4(&c[idx]) = reg_c;
    }
}

// 计算核函数运行时间的函数
float measureKernelTime(void (*kernel)(float*, float*, float*, int), float* d_a, float* d_b, float* d_c, int n, int blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 计算网格大小
    int gridSize = (n + blockSize - 1) / blockSize;

    // 记录开始时间
    cudaEventRecord(start);

    // 调用核函数
    kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算运行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// 主函数
int main() {
    const int N = 8;  // 数组大小
    const int size = N * sizeof(float);

    // 分配主机内存
    float h_a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_b[N] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float h_c[N] = {0};

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 定义线程块大小
    const int blockSize = 4;

    // 测量不同核函数的运行时间
    float time_add = measureKernelTime(add, d_a, d_b, d_c, N, blockSize);
    float time_vec2_add = measureKernelTime(vec2_add, d_a, d_b, d_c, N, blockSize);
    float time_vec4_add = measureKernelTime(vec4_add, d_a, d_b, d_c, N, blockSize);

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Result array c: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // 打印运行时间
    std::cout << "Time for add kernel: " << time_add << " ms" << std::endl;
    std::cout << "Time for vec2_add kernel: " << time_vec2_add << " ms" << std::endl;
    std::cout << "Time for vec4_add kernel: " << time_vec4_add << " ms" << std::endl;

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}