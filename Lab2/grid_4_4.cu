#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <stdio.h>
#include <math.h>
#include <string>

#define CONST_N 0.0002f
#define CONST_P 0.5f
#define CONST_G 0.75f

cudaError_t finiteElement(float* u2, float* u1, float* u, const unsigned int N, unsigned int threads_per_block, struct GpuTimer* timer, float* timeElapsed);

__global__ void finiteElementKernel(float* u2, float* u1, float* u, const unsigned int N)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    // Check for interior element
    if (idx % N != 0 && idx % N != N - 1 && idx > N && idx < N * (N - 1)) {
        u[idx] = (CONST_P * (u1[idx - 1] + u1[idx + 1] + u1[idx - N] + u1[idx + N] - 4 * u1[idx]) + 2 * u1[idx] - (1 - CONST_N) * u2[idx]) / (1 + CONST_N);
        // Boundary Conditions
        if (idx < 2 * N) {
            u[idx - N] = CONST_G * u[idx];
        }
        if (idx > N * (N - 2)) {
            u[idx + N] = CONST_G * u[idx];
            if (idx == N * (N - 2) + (N - 2)) { // Corner Element
                u[N * (N - 1) + (N - 1)] = CONST_G * u[idx + N];
            }
        }
        if (idx % N == 1) {
            u[idx - 1] = CONST_G * u[idx];
            if (idx == N * (N - 2) + 1) { // Corner Element
                u[N * (N - 1)] = CONST_G * u[idx - 1];
            }
            if (idx == N + 1) { // Corner Element
                u[0] = CONST_G * u[idx - 1];
            }
        }
        if (idx % N == N - 2) {
            u[idx + 1] = CONST_G * u[idx];
            if (idx == N - 2 + N) { // Corner Element
                u[N - 1] = CONST_G * u[idx - N];
            }
        }
    }
}


// Helper function for using CUDA to perform convolution.
cudaError_t finiteElement(float* u2, float* u1, float* u, const unsigned int N, unsigned int threads_per_block, struct GpuTimer* timer, float* timeElapsed)
{
    float* dev_u2 = 0;
    float* dev_u1 = 0;
    float* dev_u = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_u2, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_u1, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_u, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_u2, u2, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_u1, u1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_u, u, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU and time it
    timer->Start();
    finiteElementKernel <<< (N * N + threads_per_block - 1) / threads_per_block, threads_per_block >>> (dev_u2, dev_u1, dev_u, N);
    timer->Stop();

    // record the computation time
    *timeElapsed = timer->Elapsed();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "convolutionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolutionKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(u, dev_u, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_u2);
    cudaFree(dev_u1);
    cudaFree(dev_u);

    return cudaStatus;
}

int main(int argc, char** argv)
{

    if (argc != 2) {
        printf("Usage: ./sequential <num_iterations>");
        exit(1);
    }

    // load command args
    int T = std::stoi(argv[1]);

    const unsigned int N = 100;

    // Init u arrays
    float u2[N*N] = { 0 };
    float u1[N*N] = { 0 };
    float u[N*N] = { 0 };

    // Hit coordinates
    int hit_i = N/2;
    int hit_j = N/2;

    // Recording coordinates
    int rec_i = N/2;
    int rec_j = N/2;

    // Add the drum hit
    u1[hit_i * N + hit_j] = 1;

    printf("Size of grid: %d nodes\n", N*N);
    for (int k = 0; k < T; k++) {
        float timeElapsed = 0.0;
        struct GpuTimer* timer = new GpuTimer();
        finiteElement(u2, u1, u, N, 1, timer, &timeElapsed);
        // Copy elements from u to u1 and u1 to u2
        for (int i = 0; i < N * N; i++) {
                u2[i] = u1[i];
                u1[i] = u[i];
        }
        printf("(%d,%d): %f - Computation time: %f\n", rec_i, rec_j, u[rec_i * N + rec_j], timeElapsed);
        printf("\n");
    }

    return 0;
}